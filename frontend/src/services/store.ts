// Local, browser-only mock storage for users and screenings.
// Replace with a real backend when ready.
import { useEffect, useState } from "react";
import type { PredictionResult, ViewKey } from "./prediction";

export interface User { name: string; email: string }
export interface Screening extends PredictionResult { id: string; createdAt: string; images: Record<ViewKey, string>; userEmail: string }

const USERS = "scanio_users", SESSION = "scanio_session", SCANS = "scanio_scans_v2";
const read = <T,>(k: string, d: T): T => { try { return JSON.parse(localStorage.getItem(k) ?? "") as T; } catch { return d; } };
const emit = () => window.dispatchEvent(new Event("scanio-auth"));

export function register(name: string, email: string, password: string) {
  const users = read<Record<string, { name: string; password: string }>>(USERS, {});
  const key = email.toLowerCase();
  if (users[key]) throw new Error("An account with this email already exists");
  users[key] = { name, password };
  localStorage.setItem(USERS, JSON.stringify(users));
  localStorage.setItem(SESSION, JSON.stringify({ name, email: key }));
  emit();
}
export function login(email: string, password: string) {
  const users = read<Record<string, { name: string; password: string }>>(USERS, {});
  const u = users[email.toLowerCase()];
  if (!u || u.password !== password) throw new Error("Invalid email or password");
  localStorage.setItem(SESSION, JSON.stringify({ name: u.name, email: email.toLowerCase() }));
  emit();
}
export function logout() { localStorage.removeItem(SESSION); emit(); }
export function getUser(): User | null { return typeof window === "undefined" ? null : read<User | null>(SESSION, null); }

export function useUser() {
  const [user, setUser] = useState<User | null>(null);
  const [ready, setReady] = useState(false);
  useEffect(() => {
    const f = () => setUser(getUser());
    f(); setReady(true);
    window.addEventListener("scanio-auth", f);
    return () => window.removeEventListener("scanio-auth", f);
  }, []);
  return { user, ready };
}

export function saveScreening(s: Omit<Screening, "id" | "createdAt">): Screening {
  const all = read<Screening[]>(SCANS, []);
  const item = { ...s, id: crypto.randomUUID(), createdAt: new Date().toISOString() };
  try { localStorage.setItem(SCANS, JSON.stringify([item, ...all])); }
  catch { localStorage.setItem(SCANS, JSON.stringify([item, ...all.slice(0, 3)])); }
  return item;
}
export function getScreenings(email: string) { return read<Screening[]>(SCANS, []).filter((s) => s.userEmail === email); }
export function getScreening(id: string) { return read<Screening[]>(SCANS, []).find((s) => s.id === id) ?? null; }
