import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useState, type FormEvent } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { AuthShell } from "@/components/shared";
import { login } from "@/services/store";

export const Route = createFileRoute("/login")({
  head: () => ({ meta: [{ title: "Log in — Scanio AI" }, { name: "description", content: "Log in to view your dental screening history." }, { property: "og:title", content: "Log in — Scanio AI" }, { property: "og:description", content: "Access your Scanio AI dashboard." }] }),
  component: Login,
});

function Login() {
  const nav = useNavigate();
  const [email, setEmail] = useState(""), [pw, setPw] = useState(""), [err, setErr] = useState("");
  const submit = (e: FormEvent) => { e.preventDefault(); try { login(email, pw); nav({ to: "/dashboard" }); } catch (x) { setErr((x as Error).message); } };
  return (
    <AuthShell title="Welcome back" subtitle="Log in to see your screening history.">
      <form onSubmit={submit} className="space-y-4">
        <div className="space-y-1.5"><Label htmlFor="email">Email</Label><Input id="email" type="email" required value={email} onChange={(e) => setEmail(e.target.value)} /></div>
        <div className="space-y-1.5"><Label htmlFor="pw">Password</Label><Input id="pw" type="password" required value={pw} onChange={(e) => setPw(e.target.value)} /></div>
        {err && <p className="text-sm text-destructive">{err}</p>}
        <Button type="submit" className="w-full">Log in</Button>
        <p className="text-center text-sm text-muted-foreground">No account? <Link to="/register" className="font-medium text-primary">Sign up</Link></p>
      </form>
    </AuthShell>
  );
}
