import { writable } from "svelte/store";
import type { DebugDict, MoodState, EmbodiedState } from "./api";

export interface Message {
  role: "user" | "assistant";
  content: string;
  timestamp: number;
  debug?: DebugDict;
}

export const messages = writable<Message[]>([]);
export const isStreaming = writable(false);
export const sessionId = writable<string | null>(null);
export const lastDebug = writable<DebugDict | null>(null);
export const brainHistory = writable<DebugDict[]>([]);
export const mood = writable<MoodState | null>(null);
export const embodied = writable<EmbodiedState | null>(null);
export const isConnected = writable(false);
export const isEnding = writable(false);
