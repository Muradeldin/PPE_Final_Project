-- ============================================================================
-- PPE Pi: tables, photo storage and permissions
-- Run once in Supabase: SQL Editor -> New query -> paste this file -> Run.
-- Safe to run again (it skips or replaces what already exists).
--
-- Who can do what:
--   Pi (secret key)        : everything (the secret key bypasses these rules)
--   Logged-in site users   : read everything, acknowledge violations, start/stop the Pi
--   Everyone else          : nothing
-- ============================================================================

-- Violations reported by the Pi ----------------------------------------------
create table if not exists public.violations (
  id           bigint generated always as identity primary key,
  created_at   timestamptz not null default now(),
  worker_id    text        not null,
  reasons      text        not null,   -- "no_helmet", "no_vest" or "no_helmet_no_vest"
  image_path   text        not null,   -- path of the photo inside the "violations" bucket
  acknowledged boolean     not null default false
);

-- The Pi's state: what the site wants (desired_running) and what the Pi reports back
create table if not exists public.device_status (
  id              text        primary key,
  desired_running boolean     not null default false,  -- set by the site's Start/Stop buttons
  is_running      boolean     not null default false,  -- reported by the Pi
  source          text,                                 -- reported by the Pi (video file or camera)
  fps             real,                                 -- reported by the Pi
  stream_url      text,                                 -- reserved for the livestream
  last_seen       timestamptz                           -- Pi heartbeat, every ~2 s
);
insert into public.device_status (id) values ('pi') on conflict (id) do nothing;

-- Row-level security ------------------------------------------------------------
alter table public.violations    enable row level security;
alter table public.device_status enable row level security;

drop policy if exists "logged-in users read violations" on public.violations;
create policy "logged-in users read violations"
  on public.violations for select to authenticated using (true);

drop policy if exists "logged-in users acknowledge violations" on public.violations;
create policy "logged-in users acknowledge violations"
  on public.violations for update to authenticated using (true) with check (true);

drop policy if exists "logged-in users read device status" on public.device_status;
create policy "logged-in users read device status"
  on public.device_status for select to authenticated using (true);

drop policy if exists "logged-in users start and stop the pi" on public.device_status;
create policy "logged-in users start and stop the pi"
  on public.device_status for update to authenticated using (true) with check (true);

-- Site users may only change these two columns; everything else is written by the Pi
revoke insert, update, delete on public.violations    from anon, authenticated;
revoke insert, update, delete on public.device_status from anon, authenticated;
grant update (acknowledged)    on public.violations    to authenticated;
grant update (desired_running) on public.device_status to authenticated;

-- Photos: private bucket, only logged-in users can view --------------------------
insert into storage.buckets (id, name, public)
values ('violations', 'violations', false)
on conflict (id) do nothing;

drop policy if exists "logged-in users view violation photos" on storage.objects;
create policy "logged-in users view violation photos"
  on storage.objects for select to authenticated using (bucket_id = 'violations');
