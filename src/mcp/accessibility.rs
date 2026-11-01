//! Bounded AT-SPI queries run only in the MCP process, never in the compositor.

use std::collections::{HashSet, VecDeque};
use std::future::Future;
use std::time::Duration;

use anyhow::{bail, ensure, Context, Result};
use atspi::proxy::accessible::{AccessibleProxy, ObjectRefExt};
use atspi::{CoordType, Interface, ObjectRefOwned};
use futures_util::future::{select, Either};
use serde::Serialize;
use serde_json::{json, Value};
use zbus::{Connection, Proxy};

use super::{integer, string, Target};

fn check_unlocked(window: &Target) -> Result<()> {
    super::request(niri_ipc::Request::AgentInput {
        geometry: None,
        workspace: std::env::var("NIRI_MCP_WORKSPACE").ok(),
        window_id: window.id(),
        events: vec![],
    })?;
    Ok(())
}

pub(super) fn bounded<T>(future: impl Future<Output = Result<T>>) -> Result<T> {
    async_io::block_on(async {
        match select(
            Box::pin(future),
            Box::pin(async {
                let started = std::time::Instant::now();
                loop {
                    if super::cancelled() || started.elapsed() >= Duration::from_secs(10) {
                        break;
                    }
                    async_io::Timer::after(Duration::from_millis(20)).await;
                }
            }),
        )
        .await
        {
            Either::Left((result, _)) => result,
            Either::Right(_) => anyhow::bail!("accessibility operation cancelled or timed out"),
        }
    })
}

pub(super) async fn enable() -> Result<()> {
    let connection = Connection::session().await?;
    let status = Proxy::new(
        &connection,
        "org.a11y.Bus",
        "/org/a11y/bus",
        "org.a11y.Status",
    )
    .await?;
    status.set_property("IsEnabled", true).await?;
    Ok(())
}

async fn connect() -> Result<Connection> {
    let address = match std::env::var("AT_SPI_BUS_ADDRESS") {
        Ok(address) if !address.is_empty() => address,
        _ => {
            let session = Connection::session().await?;
            let bus = Proxy::new(&session, "org.a11y.Bus", "/org/a11y/bus", "org.a11y.Bus").await?;
            bus.call("GetAddress", &()).await?
        }
    };
    // Connect to the bus directly. Peer discovery can fail on unrelated sandboxed clients.
    Ok(zbus::connection::Builder::address(address.as_str())?
        .build()
        .await?)
}

async fn proxy<'a>(
    connection: &'a Connection,
    object: &ObjectRefOwned,
    interface: &'static str,
) -> Result<Proxy<'a>> {
    Ok(zbus::proxy::Builder::new(connection)
        .destination(
            object
                .name()
                .context("accessible object has no bus name")?
                .clone(),
        )?
        .path(object.path().to_owned())?
        .interface(interface)?
        .cache_properties(zbus::proxy::CacheProperties::No)
        .build()
        .await?)
}

async fn root(connection: &Connection, window: &Target) -> Result<ObjectRefOwned> {
    let pid = u32::try_from(window.pid().context("window has no process ID")?)?;
    let title = window
        .title()
        .filter(|s| !s.is_empty())
        .context("window has no title for accessibility matching")?;
    let registry = AccessibleProxy::builder(connection)
        .destination("org.a11y.atspi.Registry")?
        .cache_properties(zbus::proxy::CacheProperties::No)
        .build()
        .await?;
    let dbus = zbus::fdo::DBusProxy::new(connection).await?;
    let mut matches = Vec::new();
    for app in registry.get_children().await?.into_iter().take(256) {
        let Some(name) = app.name() else { continue };
        if dbus
            .get_connection_unix_process_id(name.clone().into())
            .await
            .ok()
            != Some(pid)
        {
            continue;
        }
        let app_proxy = app.as_accessible_proxy(connection).await?;
        for index in 0..app_proxy.child_count().await?.clamp(0, 256) {
            let candidate = app_proxy.get_child_at_index(index).await?;
            let accessible = candidate.as_accessible_proxy(connection).await?;
            if accessible.name().await.ok().as_deref() == Some(title) {
                matches.push(candidate);
            }
        }
    }
    ensure!(
        matches.len() == 1,
        "expected one accessibility window matching PID and title, found {}",
        matches.len()
    );
    Ok(matches.remove(0))
}

#[derive(Serialize)]
struct Node {
    element_index: usize,
    parent_index: Option<usize>,
    role: String,
    name: String,
    description: String,
    states: Vec<atspi::State>,
    text: Option<String>,
    bounds: Option<Value>,
    actions: Vec<String>,
    supports_editable_text: bool,
    value: Option<f64>,
    #[serde(skip)]
    reference: ObjectRefOwned,
}

async fn snapshot(
    connection: &Connection,
    root: &ObjectRefOwned,
    max_nodes: usize,
    max_depth: usize,
) -> Result<(Vec<Node>, bool)> {
    let mut queue = VecDeque::from([(root.clone(), None, 0)]);
    let mut seen = HashSet::new();
    let mut nodes = Vec::new();
    let mut truncated = false;
    while let Some((reference, parent_index, depth)) = queue.pop_front() {
        if reference.is_null() || !seen.insert(reference.clone()) {
            continue;
        }
        let accessible = reference.as_accessible_proxy(connection).await?;
        let interfaces = accessible.get_interfaces().await?;
        let (role, name, description, states, count) = futures_util::try_join!(
            accessible.get_role_name(),
            accessible.name(),
            accessible.description(),
            accessible.get_state(),
            accessible.child_count()
        )?;
        let index = nodes.len();
        let mut node = Node {
            element_index: index,
            parent_index,
            role,
            name,
            description,
            states: states.iter().collect(),
            text: None,
            bounds: None,
            actions: Vec::new(),
            supports_editable_text: interfaces.contains(Interface::EditableText),
            value: None,
            reference: reference.clone(),
        };
        if interfaces.contains(Interface::Text) {
            let text = proxy(connection, &reference, "org.a11y.atspi.Text").await?;
            let count: i32 = text.get_property("CharacterCount").await.unwrap_or(0);
            node.text = text
                .call("GetText", &(0i32, count.clamp(0, 4096)))
                .await
                .ok();
        }
        if interfaces.contains(Interface::Component) {
            let component = proxy(connection, &reference, "org.a11y.atspi.Component").await?;
            if let Ok((x, y, width, height)) = component
                .call::<_, _, (i32, i32, i32, i32)>("GetExtents", &(CoordType::Window,))
                .await
            {
                node.bounds = Some(
                    json!({"x":x,"y":y,"width":width,"height":height,"coordinate_space":"window-logical"}),
                );
            }
        }
        if interfaces.contains(Interface::Action) {
            let action = proxy(connection, &reference, "org.a11y.atspi.Action").await?;
            let actions: Vec<atspi::Action> =
                action.call("GetActions", &()).await.unwrap_or_default();
            node.actions = actions.into_iter().map(|action| action.name).collect();
        }
        if interfaces.contains(Interface::Value) {
            node.value = proxy(connection, &reference, "org.a11y.atspi.Value")
                .await?
                .get_property("CurrentValue")
                .await
                .ok();
        }
        nodes.push(node);
        let remaining = max_nodes.saturating_sub(nodes.len() + queue.len());
        let children = usize::try_from(count.max(0))?;
        let to_read = if depth < max_depth {
            children.min(remaining)
        } else {
            0
        };
        truncated |= children > to_read;
        for child in 0..to_read {
            if let Ok(child) = accessible.get_child_at_index(child as i32).await {
                queue.push_back((child, Some(index), depth + 1));
            }
        }
    }
    Ok((nodes, truncated))
}

struct Snapshot {
    id: u64,
    window_id: u64,
    root: ObjectRefOwned,
    nodes: Vec<Node>,
}

#[derive(Default)]
pub struct Accessibility {
    connection: Option<Connection>,
    current: Option<Snapshot>,
    sequence: u64,
}

impl Accessibility {
    pub fn invalidate(&mut self) {
        self.current = None;
    }

    fn connection(&mut self) -> Result<Connection> {
        if self.connection.is_none() {
            self.connection = Some(bounded(connect())?);
        }
        Ok(self.connection.as_ref().unwrap().clone())
    }

    pub fn snapshot(&mut self, window: &Target, args: &Value) -> Result<Value> {
        self.invalidate();
        let max_nodes = args
            .get("max_nodes")
            .map_or(Ok(500), |_| integer(args, "max_nodes"))?;
        let max_depth = args
            .get("max_depth")
            .map_or(Ok(32), |_| integer(args, "max_depth"))?;
        ensure!(
            (1..=2000).contains(&max_nodes) && (1..=64).contains(&max_depth),
            "invalid accessibility limits"
        );
        let connection = self.connection()?;
        let (root, nodes, truncated) = bounded(async {
            let root = root(&connection, window).await?;
            let (nodes, truncated) =
                snapshot(&connection, &root, max_nodes as usize, max_depth as usize).await?;
            Ok((root, nodes, truncated))
        })?;
        check_unlocked(window)?;
        self.sequence = self
            .sequence
            .checked_add(1)
            .context("snapshot sequence exhausted")?;
        let data = json!({"snapshot_id":self.sequence,"accessibility_tree":nodes,"accessibility_truncated":truncated});
        self.current = Some(Snapshot {
            id: self.sequence,
            window_id: window.id(),
            root,
            nodes,
        });
        Ok(data)
    }

    pub fn act(&mut self, window: &Target, tool: &str, args: &Value) -> Result<()> {
        let snapshot = self
            .current
            .take()
            .context("refresh get_app_state before accessibility actions")?;
        ensure!(
            snapshot.id == integer(args, "snapshot_id")? && snapshot.window_id == window.id(),
            "snapshot does not match the requested window"
        );
        let index = usize::try_from(integer(args, "element_index")?)?;
        let node = snapshot
            .nodes
            .get(index)
            .context("element index is not in the snapshot")?;
        let connection = self.connection()?;
        bounded(async {
            ensure!(
                root(&connection, window).await? == snapshot.root,
                "accessibility window changed, refresh get_app_state"
            );
            let accessible = node.reference.as_accessible_proxy(&connection).await?;
            ensure!(
                accessible.name().await? == node.name
                    && accessible.get_role_name().await? == node.role,
                "element changed, refresh get_app_state"
            );
            let mut ancestor = node.reference.clone();
            let mut contained = false;
            for _ in 0..=64 {
                if ancestor == snapshot.root {
                    contained = true;
                    break;
                }
                if ancestor.is_null() {
                    break;
                }
                ancestor = ancestor
                    .as_accessible_proxy(&connection)
                    .await?
                    .parent()
                    .await?;
            }
            ensure!(
                contained,
                "element no longer belongs to the requested window"
            );
            if tool == "perform_action" {
                let name = string(args, "action")?;
                let action = proxy(&connection, &node.reference, "org.a11y.atspi.Action").await?;
                let actions: Vec<atspi::Action> = action.call("GetActions", &()).await?;
                let matching: Vec<_> = actions
                    .iter()
                    .enumerate()
                    .filter(|(_, a)| a.name == name)
                    .collect();
                ensure!(
                    matching.len() == 1,
                    "action name must match exactly one available action"
                );
                check_unlocked(window)?;
                let done: bool = action.call("DoAction", &(matching[0].0 as i32,)).await?;
                ensure!(done, "application declined the accessibility action");
            } else {
                let value = string(args, "value")?;
                let interfaces = accessible.get_interfaces().await?;
                if interfaces.contains(Interface::EditableText) {
                    let editable =
                        proxy(&connection, &node.reference, "org.a11y.atspi.EditableText").await?;
                    check_unlocked(window)?;
                    let done: bool = editable.call("SetTextContents", &(value,)).await?;
                    ensure!(done, "application declined the text replacement");
                } else if interfaces.contains(Interface::Value) {
                    let value: f64 = value.parse()?;
                    ensure!(value.is_finite(), "numeric value must be finite");
                    let target =
                        proxy(&connection, &node.reference, "org.a11y.atspi.Value").await?;
                    check_unlocked(window)?;
                    target.set_property("CurrentValue", value).await?;
                } else {
                    bail!("element has no editable text or numeric value interface");
                }
            }
            Ok(())
        })
    }
}
