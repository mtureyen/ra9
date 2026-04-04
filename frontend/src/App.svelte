<script lang="ts">
  import Chat from "./lib/Chat.svelte";
  import BrainMonitor from "./lib/BrainMonitor.svelte";
  import StatusBar from "./lib/StatusBar.svelte";
  import { bootSession, endSession, getHealth } from "./lib/api";
  import { sessionId, isConnected, isEnding } from "./lib/stores";

  let status = $state<"disconnected" | "connected" | "session">("disconnected");

  async function checkConnection() {
    try {
      const health = await getHealth();
      $isConnected = health.brain;
      if (health.session) {
        status = "session";
      } else if (health.brain) {
        status = "connected";
      }
    } catch {
      $isConnected = false;
      status = "disconnected";
    }
  }

  async function handleBoot() {
    try {
      const result = await bootSession();
      $sessionId = result.session_id;
      status = "session";
    } catch (e: any) {
      alert("Failed to boot session: " + e.message);
    }
  }

  async function handleEnd() {
    $isEnding = true;
    try {
      await endSession();
      $sessionId = null;
      status = "connected";
    } catch (e: any) {
      alert("Failed to end session: " + e.message);
    } finally {
      $isEnding = false;
    }
  }

  // Check connection on mount
  $effect(() => {
    checkConnection();
  });
</script>

<div class="app">
  {#if status === "disconnected"}
    <div class="centered">
      <h1>ra9</h1>
      <p class="muted">API server not running</p>
      <p class="muted small">Start with: python -m emotive.api</p>
      <button onclick={() => checkConnection()}>Retry</button>
    </div>
  {:else if status === "connected"}
    <div class="centered">
      <h1>ra9</h1>
      <p class="muted">Brain ready. No active session.</p>
      <button class="primary" onclick={handleBoot}>Boot Session</button>
    </div>
  {:else}
    <div class="main">
      <div class="chat-panel">
        <Chat />
      </div>
      <div class="brain-panel">
        <BrainMonitor />
      </div>
    </div>
    <StatusBar onEnd={handleEnd} />
  {/if}
</div>

<style>
  .app {
    height: 100vh;
    display: flex;
    flex-direction: column;
  }

  .centered {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
  }

  .centered h1 {
    font-size: 48px;
    font-weight: 300;
    color: var(--accent);
    letter-spacing: 4px;
  }

  .muted {
    color: var(--text-secondary);
  }

  .small {
    font-size: 11px;
    color: var(--text-muted);
  }

  button {
    padding: 8px 24px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--bg-panel);
    color: var(--text-primary);
    font-family: var(--font);
    font-size: 13px;
    cursor: pointer;
    transition: all 0.15s;
  }

  button:hover {
    border-color: var(--accent);
    background: var(--bg-secondary);
  }

  button.primary {
    background: var(--accent);
    border-color: var(--accent);
    color: white;
  }

  button.primary:hover {
    background: var(--accent-dim);
  }

  .main {
    flex: 1;
    display: flex;
    overflow: hidden;
  }

  .chat-panel {
    flex: 1;
    min-width: 0;
    border-right: 1px solid var(--border);
  }

  .brain-panel {
    width: 340px;
    flex-shrink: 0;
    overflow-y: auto;
  }
</style>
