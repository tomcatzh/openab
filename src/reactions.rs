use crate::adapter::{ChatAdapter, MessageRef};
use crate::config::{ReactionEmojis, ReactionTiming};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::Duration;

const CODING_TOKENS: &[&str] = &["exec", "process", "read", "write", "edit", "bash", "shell"];
const WEB_TOKENS: &[&str] = &[
    "web_search",
    "web_fetch",
    "web-search",
    "web-fetch",
    "browser",
];

fn classify_tool<'a>(name: &str, emojis: &'a ReactionEmojis) -> &'a str {
    let n = name.to_lowercase();
    if WEB_TOKENS.iter().any(|t| n.contains(t)) {
        &emojis.web
    } else if CODING_TOKENS.iter().any(|t| n.contains(t)) {
        &emojis.coding
    } else {
        &emojis.tool
    }
}

struct Inner {
    adapter: Arc<dyn ChatAdapter>,
    // Every message that triggered this dispatch (e.g. all messages packed into
    // one batch), so the whole batch reflects the turn's current status --- not
    // just one "anchor" message that happens to be last. Previously this was a
    // single `MessageRef`, which meant only the last message in a multi-message
    // batch ever got the thinking/done/error/stall reactions; earlier messages
    // in the same batch were stuck at the initial queued emoji forever.
    messages: Vec<MessageRef>,
    emojis: ReactionEmojis,
    timing: ReactionTiming,
    current: String,
    finished: bool,
    debounce_handle: Option<tokio::task::JoinHandle<()>>,
    stall_soft_handle: Option<tokio::task::JoinHandle<()>>,
    stall_hard_handle: Option<tokio::task::JoinHandle<()>>,
}

pub struct StatusReactionController {
    inner: Arc<Mutex<Inner>>,
    enabled: bool,
}

/// Add `emoji` to every message, then (if `old` is set and different) remove
/// `old` from every message. Best-effort: a failure on one message does not
/// stop the others (each call result is ignored, matching prior behavior).
async fn apply_to_all(
    adapter: &Arc<dyn ChatAdapter>,
    messages: &[MessageRef],
    old: &str,
    new: &str,
) {
    for msg in messages {
        let _ = adapter.add_reaction(msg, new).await;
    }
    if !old.is_empty() && old != new {
        for msg in messages {
            let _ = adapter.remove_reaction(msg, old).await;
        }
    }
}

impl StatusReactionController {
    pub fn new(
        enabled: bool,
        adapter: Arc<dyn ChatAdapter>,
        messages: Vec<MessageRef>,
        emojis: ReactionEmojis,
        timing: ReactionTiming,
    ) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                adapter,
                messages,
                emojis,
                timing,
                current: String::new(),
                finished: false,
                debounce_handle: None,
                stall_soft_handle: None,
                stall_hard_handle: None,
            })),
            enabled,
        }
    }

    pub async fn set_queued(&self) {
        if !self.enabled {
            return;
        }
        let emoji = { self.inner.lock().await.emojis.queued.clone() };
        self.apply_immediate(&emoji).await;
    }

    pub async fn set_thinking(&self) {
        if !self.enabled {
            return;
        }
        let emoji = { self.inner.lock().await.emojis.thinking.clone() };
        self.schedule_debounced(&emoji).await;
    }

    pub async fn set_tool(&self, tool_name: &str) {
        if !self.enabled {
            return;
        }
        let emoji = {
            let inner = self.inner.lock().await;
            classify_tool(tool_name, &inner.emojis).to_string()
        };
        self.schedule_debounced(&emoji).await;
    }

    pub async fn set_done(&self) {
        if !self.enabled {
            return;
        }
        let emoji = { self.inner.lock().await.emojis.done.clone() };
        self.finish(&emoji).await;
        // Add a random mood face to every tracked message.
        let faces = ["😊", "😎", "🫡", "🤓", "😏", "✌️", "💪", "🦾"];
        let face = faces[rand::random::<usize>() % faces.len()];
        let inner = self.inner.lock().await;
        for msg in &inner.messages {
            let _ = inner.adapter.add_reaction(msg, face).await;
        }
    }

    pub async fn set_error(&self) {
        if !self.enabled {
            return;
        }
        let emoji = { self.inner.lock().await.emojis.error.clone() };
        self.finish(&emoji).await;
    }

    pub async fn clear(&self) {
        if !self.enabled {
            return;
        }
        let mut inner = self.inner.lock().await;
        cancel_timers(&mut inner);
        let current = inner.current.clone();
        if !current.is_empty() {
            for msg in &inner.messages {
                let _ = inner.adapter.remove_reaction(msg, &current).await;
            }
            inner.current.clear();
        }
    }

    async fn apply_immediate(&self, emoji: &str) {
        let mut inner = self.inner.lock().await;
        if inner.finished || emoji == inner.current {
            return;
        }
        cancel_debounce(&mut inner);
        let old = inner.current.clone();
        inner.current = emoji.to_string();
        let adapter = inner.adapter.clone();
        let messages = inner.messages.clone();
        let new = emoji.to_string();
        drop(inner);

        apply_to_all(&adapter, &messages, &old, &new).await;
        self.reset_stall_timers().await;
    }

    async fn schedule_debounced(&self, emoji: &str) {
        let mut inner = self.inner.lock().await;
        if inner.finished || emoji == inner.current {
            self.reset_stall_timers_inner(&mut inner);
            return;
        }
        cancel_debounce(&mut inner);

        let emoji = emoji.to_string();
        let ctrl = self.inner.clone();
        let debounce_ms = inner.timing.debounce_ms;
        inner.debounce_handle = Some(tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(debounce_ms)).await;
            let mut inner = ctrl.lock().await;
            if inner.finished {
                return;
            }
            let old = inner.current.clone();
            inner.current = emoji.clone();
            let adapter = inner.adapter.clone();
            let messages = inner.messages.clone();
            drop(inner);

            apply_to_all(&adapter, &messages, &old, &emoji).await;
        }));
        self.reset_stall_timers_inner(&mut inner);
    }

    async fn finish(&self, emoji: &str) {
        let mut inner = self.inner.lock().await;
        if inner.finished {
            return;
        }
        inner.finished = true;
        cancel_timers(&mut inner);

        let old = inner.current.clone();
        inner.current = emoji.to_string();
        let adapter = inner.adapter.clone();
        let messages = inner.messages.clone();
        let new = emoji.to_string();
        drop(inner);

        apply_to_all(&adapter, &messages, &old, &new).await;
    }

    async fn reset_stall_timers(&self) {
        let mut inner = self.inner.lock().await;
        self.reset_stall_timers_inner(&mut inner);
    }

    fn reset_stall_timers_inner(&self, inner: &mut Inner) {
        if let Some(h) = inner.stall_soft_handle.take() {
            h.abort();
        }
        if let Some(h) = inner.stall_hard_handle.take() {
            h.abort();
        }

        let soft_ms = inner.timing.stall_soft_ms;
        let hard_ms = inner.timing.stall_hard_ms;
        let ctrl = self.inner.clone();

        inner.stall_soft_handle = Some(tokio::spawn({
            let ctrl = ctrl.clone();
            async move {
                tokio::time::sleep(Duration::from_millis(soft_ms)).await;
                let mut inner = ctrl.lock().await;
                if inner.finished {
                    return;
                }
                let old = inner.current.clone();
                inner.current = "🥱".to_string();
                let adapter = inner.adapter.clone();
                let messages = inner.messages.clone();
                drop(inner);
                apply_to_all(&adapter, &messages, &old, "🥱").await;
            }
        }));

        inner.stall_hard_handle = Some(tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(hard_ms)).await;
            let mut inner = ctrl.lock().await;
            if inner.finished {
                return;
            }
            let old = inner.current.clone();
            inner.current = "😨".to_string();
            let adapter = inner.adapter.clone();
            let messages = inner.messages.clone();
            drop(inner);
            apply_to_all(&adapter, &messages, &old, "😨").await;
        }));
    }
}

fn cancel_debounce(inner: &mut Inner) {
    if let Some(h) = inner.debounce_handle.take() {
        h.abort();
    }
}

fn cancel_timers(inner: &mut Inner) {
    if let Some(h) = inner.debounce_handle.take() {
        h.abort();
    }
    if let Some(h) = inner.stall_soft_handle.take() {
        h.abort();
    }
    if let Some(h) = inner.stall_hard_handle.take() {
        h.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::{ChannelRef, ChatAdapter};
    use anyhow::Result;
    use async_trait::async_trait;

    /// Records every add_reaction/remove_reaction call as (message_id, emoji, is_add)
    /// so tests can assert exactly which messages were touched.
    #[derive(Clone, Default)]
    struct RecordingAdapter {
        calls: Arc<std::sync::Mutex<Vec<(String, String, bool)>>>,
    }

    impl RecordingAdapter {
        fn calls_for(&self, message_id: &str) -> Vec<(String, bool)> {
            self.calls
                .lock()
                .unwrap()
                .iter()
                .filter(|(id, _, _)| id == message_id)
                .map(|(_, emoji, is_add)| (emoji.clone(), *is_add))
                .collect()
        }
    }

    #[async_trait]
    impl ChatAdapter for RecordingAdapter {
        fn platform(&self) -> &'static str {
            "test"
        }
        fn message_limit(&self) -> usize {
            2000
        }
        async fn send_message(&self, _: &ChannelRef, _: &str) -> Result<MessageRef> {
            unimplemented!()
        }
        async fn create_thread(
            &self,
            _: &ChannelRef,
            _: &MessageRef,
            _: &str,
        ) -> Result<ChannelRef> {
            unimplemented!()
        }
        async fn add_reaction(&self, msg: &MessageRef, emoji: &str) -> Result<()> {
            self.calls
                .lock()
                .unwrap()
                .push((msg.message_id.clone(), emoji.to_string(), true));
            Ok(())
        }
        async fn remove_reaction(&self, msg: &MessageRef, emoji: &str) -> Result<()> {
            self.calls
                .lock()
                .unwrap()
                .push((msg.message_id.clone(), emoji.to_string(), false));
            Ok(())
        }
        fn use_streaming(&self, _other_bot_present: bool) -> bool {
            true
        }
    }

    fn msg(id: &str) -> MessageRef {
        MessageRef {
            channel: ChannelRef {
                platform: "test".into(),
                channel_id: "chan".into(),
                thread_id: None,
                parent_id: None,
                origin_event_id: None,
            },
            message_id: id.to_string(),
        }
    }

    fn default_emojis() -> ReactionEmojis {
        serde_json::from_value(serde_json::json!({})).unwrap()
    }

    fn default_timing() -> ReactionTiming {
        serde_json::from_value(serde_json::json!({})).unwrap()
    }

    fn controller(adapter: RecordingAdapter, messages: Vec<MessageRef>) -> StatusReactionController {
        StatusReactionController::new(
            true,
            Arc::new(adapter),
            messages,
            default_emojis(),
            default_timing(),
        )
    }

    #[tokio::test]
    async fn set_queued_applies_to_every_message_in_a_batch() {
        let adapter = RecordingAdapter::default();
        let ctrl = controller(adapter.clone(), vec![msg("a"), msg("b"), msg("c")]);
        ctrl.set_queued().await;
        for id in ["a", "b", "c"] {
            assert_eq!(
                adapter.calls_for(id),
                vec![("👀".to_string(), true)],
                "message {id} should have gotten the queued reaction"
            );
        }
    }

    #[tokio::test]
    async fn set_done_applies_to_every_message_not_just_the_last() {
        // Regression test: previously only the LAST message in a multi-message
        // batch was tracked, so earlier messages got the initial queued emoji
        // and were never updated again (stuck forever, even after the turn
        // finished). Every message in the batch must reach "done".
        let adapter = RecordingAdapter::default();
        let ctrl = controller(adapter.clone(), vec![msg("first"), msg("second")]);
        ctrl.set_queued().await;
        ctrl.set_done().await;
        for id in ["first", "second"] {
            let calls = adapter.calls_for(id);
            assert!(
                calls.iter().any(|(emoji, is_add)| emoji == "🆗" && *is_add),
                "message {id} should have gotten the done reaction, got {calls:?}"
            );
            // queued (👀) must have been removed once done was applied.
            assert!(
                calls.iter().any(|(emoji, is_add)| emoji == "👀" && !is_add),
                "message {id} should have had the queued reaction removed, got {calls:?}"
            );
        }
    }

    #[tokio::test]
    async fn set_error_applies_to_every_message_in_a_batch() {
        let adapter = RecordingAdapter::default();
        let ctrl = controller(adapter.clone(), vec![msg("x"), msg("y")]);
        ctrl.set_queued().await;
        ctrl.set_error().await;
        for id in ["x", "y"] {
            let calls = adapter.calls_for(id);
            assert!(
                calls.iter().any(|(emoji, is_add)| emoji == "😱" && *is_add),
                "message {id} should have gotten the error reaction, got {calls:?}"
            );
        }
    }

    #[tokio::test]
    async fn clear_removes_from_every_message_in_a_batch() {
        let adapter = RecordingAdapter::default();
        let ctrl = controller(adapter.clone(), vec![msg("p"), msg("q")]);
        ctrl.set_queued().await;
        ctrl.clear().await;
        for id in ["p", "q"] {
            let calls = adapter.calls_for(id);
            assert!(
                calls.iter().any(|(emoji, is_add)| emoji == "👀" && !is_add),
                "message {id} should have had its reaction cleared, got {calls:?}"
            );
        }
    }

    #[tokio::test]
    async fn single_message_batch_still_works() {
        // The adapter.rs (non-batched) call site passes a one-element Vec;
        // confirm the single-message path is unaffected by the refactor.
        let adapter = RecordingAdapter::default();
        let ctrl = controller(adapter.clone(), vec![msg("solo")]);
        ctrl.set_queued().await;
        ctrl.set_done().await;
        let calls = adapter.calls_for("solo");
        assert!(calls.iter().any(|(emoji, is_add)| emoji == "🆗" && *is_add));
    }
}
