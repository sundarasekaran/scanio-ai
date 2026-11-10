import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useState, type FormEvent } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { AuthShell } from "@/components/shared";
import { register } from "@/services/store";

export const Route = createFileRoute("/register")({
  head: () => ({ meta: [{ title: "Create account — Scanio AI" }, { name: "description", content: "Create a free Scanio AI account to save your dental screenings." }, { property: "og:title", content: "Create account — Scanio AI" }, { property: "og:description", content: "Sign up to save screenings and download reports." }] }),
  component: Register,
});

function Register() {
  const nav = useNavigate();
  const [name, setName] = useState(""), [email, setEmail] = useState(""), [pw, setPw] = useState(""), [err, setErr] = useState("");
  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (pw.length < 6) return setErr("Password must be at least 6 characters");
    try { register(name, email, pw); nav({ to: "/dashboard" }); } catch (x) { setErr((x as Error).message); }
  };
  return (
    <AuthShell title="Create your account" subtitle="Save screenings and track your oral health.">
      <form onSubmit={submit} className="space-y-4">
        <div className="space-y-1.5"><Label htmlFor="name">Full name</Label><Input id="name" required value={name} onChange={(e) => setName(e.target.value)} /></div>
        <div className="space-y-1.5"><Label htmlFor="email">Email</Label><Input id="email" type="email" required value={email} onChange={(e) => setEmail(e.target.value)} /></div>
        <div className="space-y-1.5"><Label htmlFor="pw">Password</Label><Input id="pw" type="password" required value={pw} onChange={(e) => setPw(e.target.value)} /></div>
        {err && <p className="text-sm text-destructive">{err}</p>}
        <Button type="submit" className="w-full">Create account</Button>
        <p className="text-center text-sm text-muted-foreground">Already registered? <Link to="/login" className="font-medium text-primary">Log in</Link></p>
      </form>
    </AuthShell>
  );
}
