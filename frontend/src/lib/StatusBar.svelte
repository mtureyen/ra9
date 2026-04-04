<script lang="ts">
  import { sessionId, mood, embodied, isEnding, messages } from "./stores";

  interface Props {
    onEnd: () => void;
  }

  let { onEnd }: Props = $props();

  function moodLabel(dim: string, val: number): string {
    const dev = val - 0.5;
    if (Math.abs(dev) < 0.05) return "";
    return dev > 0 ? "↑" : "↓";
  }

  function moodColor(val: number): string {
    const dev = Math.abs(val - 0.5);
    if (dev < 0.05) return "var(--text-muted)";
    if (dev < 0.15) return "var(--text-secondary)";
    return val > 0.5 ? "var(--success)" : "var(--warning)";
  }
</script>

<div class="status-bar">
  <div class="left">
    {#if $mood}
      <div class="mood-dims">
        {#each Object.entries($mood) as [dim, val]}
          <span class="dim" style="color: {moodColor(val)}" title="{dim}: {val.toFixed(2)}">
            {dim.replace("_", " ").slice(0, 3)}{moodLabel(dim, val)}
          </span>
        {/each}
      </div>
    {/if}
  </div>

  <div class="center">
    {#if $embodied}
      <span class="energy" title="Energy: {($embodied.energy * 100).toFixed(0)}%">
        ⚡ {($embodied.energy * 100).toFixed(0)}%
      </span>
    {/if}
    <span class="count">{$messages.length} exchanges</span>
  </div>

  <div class="right">
    <span class="session-id" title={$sessionId || ""}>
      {$sessionId ? $sessionId.slice(0, 8) + "..." : "—"}
    </span>
    <button class="end-btn" onclick={onEnd} disabled={$isEnding}>
      {$isEnding ? "Ending..." : "End Session"}
    </button>
  </div>
</div>

<style>
  .status-bar {
    height: 32px;
    display: flex;
    align-items: center;
    padding: 0 12px;
    background: var(--bg-secondary);
    border-top: 1px solid var(--border);
    font-size: 11px;
    gap: 16px;
  }

  .left, .center, .right {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .left { flex: 1; }
  .center { flex: 0; white-space: nowrap; }
  .right { flex: 1; justify-content: flex-end; }

  .mood-dims {
    display: flex;
    gap: 6px;
  }

  .dim {
    font-size: 10px;
    font-weight: 500;
  }

  .energy {
    color: var(--warning);
  }

  .count {
    color: var(--text-muted);
  }

  .session-id {
    color: var(--text-muted);
    font-size: 10px;
  }

  .end-btn {
    padding: 2px 12px;
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text-secondary);
    font-family: var(--font);
    font-size: 11px;
    cursor: pointer;
  }

  .end-btn:hover:not(:disabled) {
    border-color: var(--danger);
    color: var(--danger);
  }

  .end-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
</style>
