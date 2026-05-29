-- ============================================================
-- Receipt Intelligence — Supabase PostgreSQL Schema
-- Run this in your Supabase project's SQL Editor
-- ============================================================

-- Users are handled by Supabase Auth (auth.users table)
-- Profiles table links to auth.users and stores the username

create table public.profiles (
    id          uuid primary key references auth.users(id) on delete cascade,
    username    text unique not null,
    created_at  timestamptz default now()
);

create table public.receipts (
    id             uuid primary key default gen_random_uuid(),
    user_id        uuid not null references public.profiles(id) on delete cascade,
    store_name     text,
    store_location text,
    receipt_total  numeric(10,2) default 0,
    receipt_no     text,
    kra_pin        text,
    cashier_name   text,
    receipt_date   date,
    image_filename text,
    created_at     timestamptz default now()
);

create table public.items (
    id          uuid primary key default gen_random_uuid(),
    receipt_id  uuid not null references public.receipts(id) on delete cascade,
    user_id     uuid not null references public.profiles(id) on delete cascade,
    item_name   text,
    qty         numeric(10,3) default 0,
    unit_price  numeric(10,2) default 0,
    line_total  numeric(10,2) default 0,
    category    text,
    created_at  timestamptz default now()
);

-- ── Row Level Security ──────────────────────────────────────
-- Each user can only read and write their own rows.

alter table public.profiles enable row level security;
alter table public.receipts  enable row level security;
alter table public.items     enable row level security;

create policy "profiles: own row"
    on public.profiles for all
    using (auth.uid() = id);

create policy "receipts: own rows"
    on public.receipts for all
    using (auth.uid() = user_id);

create policy "items: own rows"
    on public.items for all
    using (auth.uid() = user_id);

-- ── Indexes ─────────────────────────────────────────────────
create index idx_receipts_user_id    on public.receipts(user_id);
create index idx_receipts_date       on public.receipts(receipt_date);
create index idx_items_user_id       on public.items(user_id);
create index idx_items_receipt_id    on public.items(receipt_id);
create index idx_items_category      on public.items(category);
