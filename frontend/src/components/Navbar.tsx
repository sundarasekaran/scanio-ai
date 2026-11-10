import { Link, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { Menu, ScanLine, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { logout, useUser } from "@/services/store";

export function Logo() {
  return (
    <Link to="/" className="flex items-center gap-2 font-display text-xl font-bold text-foreground">
      <span className="grid h-9 w-9 place-items-center rounded-xl bg-gradient-primary text-primary-foreground"><ScanLine className="h-5 w-5" /></span>
      Scanio <span className="text-primary">AI</span>
    </Link>
  );
}

export function Navbar() {
  const { user } = useUser();
  const [open, setOpen] = useState(false);
  const navigate = useNavigate();
  const links = [{ to: "/", label: "Home" }, { to: "/upload", label: "Screening" }, { to: "/dashboard", label: "Dashboard" }] as const;
  const out = () => { logout(); navigate({ to: "/" }); };
  return (
    <header className="sticky top-0 z-40 border-b border-border bg-background/90 backdrop-blur">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4">
        <Logo />
        <nav className="hidden items-center gap-8 md:flex">
          {links.map((l) => (
            <Link key={l.to} to={l.to} className="text-sm font-medium text-muted-foreground hover:text-primary" activeProps={{ className: "text-primary" }} activeOptions={{ exact: true }}>{l.label}</Link>
          ))}
        </nav>
        <div className="hidden items-center gap-2 md:flex">
          {user ? (<><span className="text-sm text-muted-foreground">Hi, {user.name.split(" ")[0]}</span><Button variant="outline" size="sm" onClick={out}>Log out</Button></>) : (
            <><Button variant="ghost" size="sm" asChild><Link to="/login">Log in</Link></Button><Button size="sm" asChild><Link to="/register">Sign up</Link></Button></>
          )}
        </div>
        <button className="md:hidden" onClick={() => setOpen(!open)} aria-label="Menu">{open ? <X /> : <Menu />}</button>
      </div>
      {open && (
        <div className="space-y-1 border-t border-border px-4 py-3 md:hidden">
          {links.map((l) => <Link key={l.to} to={l.to} onClick={() => setOpen(false)} className="block rounded-lg px-3 py-2 text-sm font-medium hover:bg-accent">{l.label}</Link>)}
          {user ? <Button variant="outline" className="w-full" onClick={() => { setOpen(false); out(); }}>Log out</Button> : (
            <div className="flex gap-2 pt-2"><Button variant="outline" className="flex-1" asChild><Link to="/login" onClick={() => setOpen(false)}>Log in</Link></Button><Button className="flex-1" asChild><Link to="/register" onClick={() => setOpen(false)}>Sign up</Link></Button></div>
          )}
        </div>
      )}
    </header>
  );
}
