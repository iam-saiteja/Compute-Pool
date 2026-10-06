use anyhow::{Context, Result};
use clap::{Args, CommandFactory, Parser, Subcommand};
use colored::*;
use comfy_table::{presets::UTF8_FULL, Cell, Color, ContentArrangement, Table};
use compute_pool_core::{
    auth::{
        cluster_slots, load_credentials, load_settings, save_credentials, save_settings,
        validate_web_password, Credentials, MAX_ACCOUNT_SLOTS,
    },
    job::{Job, JobSpec, JobState},
    kaggle::KaggleClient,
    probe::run_probe,
    scheduler::{get_all_statuses, schedule_job},
    shell::{launch_cluster_shell, launch_gpu_shell, stop_gpu_shell},
    storage::{
        clear_jobs, delete_job, get_job, load_all_jobs, next_job_id, reindex_jobs, upsert_job,
    },
};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
#[command(
    name = "compute-pool",
    about = "Compute Pool -- pool several Kaggle accounts' GPUs into one cluster",
    long_about = "Compute Pool launches Kaggle GPU sessions across one or more accounts, \
bridges them into a cluster you can dispatch work to, and gives you the tools to run \
independent, data-parallel or model-parallel workloads across them.\n\n\
Typical first session:\n  \
compute-pool login --slot 1          # store account 1 (asks once to accept the policy risk)\n  \
compute-pool login --slot 2          # store account 2\n  \
compute-pool pwd                     # set the web terminal / file manager password\n  \
compute-pool shell --slot cluster    # launch a 2-node cluster and print its URLs\n\n\
Account policy: Kaggle's terms are one account per person. Pooling accounts can get \
them banned. Read https://github.com/iam-saiteja/Compute-Pool/issues/12 before using this.",
    after_help = "Run `compute-pool <command> --help` for the options of a single command.",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Store a Kaggle account's username and API key for one slot (1 to 8)
    Login {
        /// Account slot, 1 to 8. Omit it to use the next free slot.
        #[arg(short, long)]
        slot: Option<usize>,

        /// Kaggle username (prompted if omitted)
        #[arg(short, long)]
        username: Option<String>,

        /// Kaggle API key (prompted, masked, if omitted)
        #[arg(short, long)]
        key: Option<String>,
    },

    /// Set or change the password for the web terminal and file manager
    #[command(name = "pwd")]
    Pwd,

    /// Choose which configured accounts make up the cluster (default: all of them)
    Cluster {
        #[command(subcommand)]
        sub: Option<ClusterSubcommand>,
    },

    /// Manage multi-account pool and inspect GPU quota
    Accounts {
        #[command(subcommand)]
        sub: AccountsSubcommand,
    },

    /// Manage batch GPU jobs (list, submit, status, logs, stop, delete, clear, reindex)
    Jobs {
        #[command(subcommand)]
        sub: JobsSubcommand,
    },

    /// Probe live GPU hardware details on an account slot
    Probe {
        /// Account slot number, 1 to 8 (1 or 2 for a single session)
        #[arg(short, long, default_value = "1")]
        slot: usize,
    },

    /// Launch a live GPU web terminal: one account, or a cluster of the accounts chosen with `cluster`
    Shell {
        /// An account slot number, or 'cluster' (the default) for the accounts chosen with `compute-pool cluster`
        #[arg(short, long, default_value = "cluster")]
        slot: String,

        /// Session duration in minutes (max 120)
        #[arg(short, long, default_value = "120")]
        duration: u32,

        /// Automatically open Master Web Terminal in default browser
        #[arg(long)]
        open: bool,

        /// Seconds to wait for the cluster to come online before giving up (default 900)
        #[arg(long)]
        timeout: Option<u64>,
    },

    /// Stop active interactive GPU terminals and release GPU resources immediately
    #[command(name = "shell-stop")]
    ShellStop {
        /// Account slot to stop (1, 2, or leave empty for all)
        #[arg(short, long)]
        slot: Option<usize>,
    },

}

#[derive(Subcommand, Debug)]
enum ClusterSubcommand {
    /// Show which accounts the next cluster uses (also the default with no subcommand)
    Show,
    /// Use only these accounts, e.g. `cluster use 1 3`. The first one is the master.
    Use {
        /// Account slots, 1 to 8
        #[arg(required = true, value_name = "SLOT")]
        slots: Vec<usize>,
    },
    /// Use every configured account again
    Reset,
}

#[derive(Subcommand, Debug)]
enum AccountsSubcommand {
    /// List configured account slots
    List,
    /// Fetch live remaining GPU & TPU quotas
    Status,
}

#[derive(Subcommand, Debug)]
enum JobsSubcommand {
    /// List all submitted jobs
    List,
    /// Submit a Python script for GPU execution
    Submit(SubmitArgs),
    /// Inspect execution status and metadata of a job
    Status {
        /// Job ID (e.g. 0, 1, job-0)
        job_id: String,
    },
    /// View real-time logs and output of a job
    Logs {
        /// Job ID
        job_id: String,
    },
    /// Cancel and terminate a running job
    Stop {
        /// Job ID
        job_id: String,
    },
    /// Delete a job and its records permanently
    Delete {
        /// Job ID
        job_id: String,
    },
    /// Clear and delete all job records and history
    Clear {
        /// Bypass confirmation prompt
        #[arg(short, long)]
        force: bool,
    },
    /// Re-index all existing jobs sequentially starting from 0 (0, 1, 2, ...)
    Reindex,
}

#[derive(Args, Debug)]
struct SubmitArgs {
    /// Path to Python script file
    #[arg(short, long)]
    script: Option<PathBuf>,

    /// Inline Python code string
    #[arg(short, long)]
    code: Option<String>,

    /// Custom job name
    #[arg(short, long, default_value = "gpu-job")]
    name: String,

    /// Request dedicated GPU acceleration
    #[arg(long, default_value = "true")]
    gpu: bool,

    /// Max runtime hours before auto-terminating
    #[arg(short, long, default_value = "2.0")]
    runtime_hours: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // If invoked with subcommands, run directly in batch mode
    if args.len() > 1 {
        let cli = Cli::parse();
        if let Some(cmd) = cli.command {
            if let Err(e) = execute_command(cmd).await {
                eprintln!("{} Error: {:#}", "x".red().bold(), e);
                std::process::exit(1);
            }
        }
        return Ok(());
    }

    // Otherwise, launch the Interactive Terminal (REPL)
    run_interactive_terminal().await?;
    Ok(())
}

fn print_welcome_banner() {
    println!();
    println!("{}", "╭────────────────────────────────────────────────────────────────────────╮".cyan());
    println!("{}", "│                     ⚡ Compute Pool GPU Terminal ⚡                      │".bold().cyan());
    println!("{}", "│                   Native GPU Task Scheduling Console                   │".dimmed().cyan());
    println!("{}", "╰────────────────────────────────────────────────────────────────────────╯".cyan());
    println!();
    println!("{}", "Welcome to Compute Pool Interactive Terminal!".bold());
    println!("Type a command without any prefix.");
    println!();
    print_command_list();
    println!();
}

async fn run_interactive_terminal() -> Result<()> {
    print_welcome_banner();

    let mut rl = DefaultEditor::new()?;
    let history_path = dirs::home_dir().map(|h| h.join(".compute_pool_history"));

    if let Some(ref path) = history_path {
        let _ = rl.load_history(path);
    }

    loop {
        let readline = rl.readline("compute-pool > ");
        match readline {
            Ok(line) => {
                let input = line.trim();
                if input.is_empty() {
                    continue;
                }

                let _ = rl.add_history_entry(input);

                // Handle internal terminal commands
                if input.eq_ignore_ascii_case("exit")
                    || input.eq_ignore_ascii_case("quit")
                    || input.eq_ignore_ascii_case("q")
                {
                    println!("{}", "Goodbye!".green());
                    break;
                }

                if input.eq_ignore_ascii_case("cls") || input.eq_ignore_ascii_case("clear") {
                    print!("{esc}[2J{esc}[1;1H", esc = 27 as char);
                    print_welcome_banner();
                    continue;
                }

                if input.eq_ignore_ascii_case("help") || input == "?" {
                    print_help();
                    continue;
                }

                // `help login` is the same as `login --help`.
                let help_form = input.strip_prefix("help ").map(|rest| format!("{} --help", rest.trim()));
                let input = help_form.as_deref().unwrap_or(input);

                // Strip leading 'compute-pool' if typed by habit
                let clean_input = if input.starts_with("compute-pool ") {
                    input["compute-pool ".len()..].trim()
                } else if input.starts_with("compute-pool.exe ") {
                    input["compute-pool.exe ".len()..].trim()
                } else {
                    input
                };

                // Parse command arguments using shell_words
                let tokens = match shell_words::split(clean_input) {
                    Ok(t) => t,
                    Err(e) => {
                        println!("{} Invalid command line syntax: {}", "x".red().bold(), e);
                        continue;
                    }
                };

                let mut cmd_args = vec!["compute-pool".to_string()];
                cmd_args.extend(tokens);

                match Cli::try_parse_from(&cmd_args) {
                    Ok(cli) => {
                        if let Some(cmd) = cli.command {
                            if let Err(e) = execute_command(cmd).await {
                                println!("{} Error: {:#}", "x".red().bold(), e);
                            }
                        }
                    }
                    Err(e) => {
                        println!("{}", e);
                    }
                }
            }
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => {
                println!("\n{}", "Exiting Compute Pool Terminal...".dimmed());
                break;
            }
            Err(err) => {
                println!("{} Terminal error: {:?}", "x".red().bold(), err);
                break;
            }
        }
    }

    if let Some(ref path) = history_path {
        let _ = rl.save_history(path);
    }

    Ok(())
}

/// Every command, generated from the clap definitions, so this list cannot drift
/// from what the commands actually accept.
fn print_command_list() {
    let root = Cli::command();
    println!("{}", "Commands:".bold());
    for sub in root.get_subcommands() {
        let name = sub.get_name();
        if sub.get_subcommands().next().is_some() {
            for nested in sub.get_subcommands() {
                let full = format!("{} {}", name, nested.get_name());
                let about = nested.get_about().map(|a| a.to_string()).unwrap_or_default();
                println!("  {} {}", format!("{:<24}", full).bold().green(), about);
            }
        } else {
            let about = sub.get_about().map(|a| a.to_string()).unwrap_or_default();
            println!("  {} {}", format!("{:<24}", name).bold().green(), about);
        }
    }
    println!("  {} {}", format!("{:<24}", "help [command]").bold().yellow(), "Show this list, or the options of one command");
    println!("  {} {}", format!("{:<24}", "cls").bold().yellow(), "Clear the screen");
    println!("  {} {}", format!("{:<24}", "exit / quit").bold().red(), "Leave the terminal");
    println!("Every command also takes -h or --help for its own options, e.g. `login -h`.");
}

fn print_help() {
    println!();
    print_command_list();
    println!();
}

async fn execute_command(command: Commands) -> Result<()> {
    match command {
        Commands::Login { slot, username, key } => {
            handle_login(slot, username, key).await?;
        }
        Commands::Pwd => {
            set_web_password()?;
        }
        Commands::Accounts { sub } => match sub {
            AccountsSubcommand::List => handle_accounts_list().await?,
            AccountsSubcommand::Status => handle_accounts_status().await?,
        },
        Commands::Jobs { sub } => match sub {
            JobsSubcommand::List => handle_jobs_list().await?,
            JobsSubcommand::Submit(args) => handle_jobs_submit(args).await?,
            JobsSubcommand::Status { job_id } => handle_jobs_status(&job_id).await?,
            JobsSubcommand::Logs { job_id } => handle_jobs_logs(&job_id).await?,
            JobsSubcommand::Stop { job_id } => handle_jobs_stop(&job_id).await?,
            JobsSubcommand::Delete { job_id } => handle_jobs_delete(&job_id)?,
            JobsSubcommand::Clear { force } => handle_jobs_clear(force)?,
            JobsSubcommand::Reindex => handle_jobs_reindex()?,
        },
        Commands::Probe { slot } => {
            handle_probe(slot).await?;
        }
        Commands::Shell {
            slot,
            duration,
            open,
            timeout,
        } => {
            handle_shell(&slot, duration, open, timeout).await?;
        }
        Commands::Cluster { sub } => {
            handle_cluster(sub.unwrap_or(ClusterSubcommand::Show))?;
        }
        Commands::ShellStop { slot } => {
            handle_shell_stop(slot).await?;
        }
    }
    Ok(())
}

fn prompt_masked_password(prompt: &str) -> Result<String> {
    let term = console::Term::stderr();
    term.write_str(&format!("{}: ", prompt))?;
    let mut input = String::new();
    loop {
        let key = term.read_key()?;
        match key {
            console::Key::Enter => {
                term.write_line("")?;
                break;
            }
            console::Key::Backspace => {
                if !input.is_empty() {
                    input.pop();
                    term.clear_chars(1)?;
                }
            }
            console::Key::Char(c) => {
                if !c.is_control() {
                    input.push(c);
                    term.write_str("*")?;
                }
            }
            _ => {}
        }
    }
    if input.trim().is_empty() {
        anyhow::bail!("API key / token cannot be empty");
    }
    Ok(input.trim().to_string())
}

/// Asked once, before anything is stored or launched. Compute Pool pools several
/// Kaggle accounts, and Kaggle's policy is one account per person, so the user
/// has to accept that risk first. Declining stores nothing and runs nothing.
fn ensure_policy_accepted() -> Result<()> {
    let mut settings = load_settings()?;
    if settings.risk_acknowledged {
        return Ok(());
    }
    println!("\n{}", "Before you continue".bold().yellow());
    println!("Compute Pool pools several Kaggle accounts into one cluster.");
    println!("Kaggle's policy, as stated in its rules and community posts, is one account per person.");
    println!("Accounts used this way can be banned, and that applies to every account in the pool.");
    println!("I could not read Kaggle's full terms directly: check the current terms yourself.");
    println!("Research: {}", "https://github.com/iam-saiteja/Compute-Pool/issues/12".cyan());
    let accepted = dialoguer::Confirm::new()
        .with_prompt("By continuing you agree to Kaggle's one-account policy and accept the ban risk. Accept?")
        .default(false)
        .interact()?;
    if !accepted {
        anyhow::bail!("Not accepted. Nothing was stored or launched.");
    }
    settings.risk_acknowledged = true;
    save_settings(&settings)?;
    Ok(())
}

fn join_slots(slots: &[usize]) -> String {
    slots.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", ")
}

fn configured_slots() -> Result<Vec<usize>> {
    let mut configured = Vec::new();
    for slot in 1..=MAX_ACCOUNT_SLOTS {
        if load_credentials(slot)?.is_some() {
            configured.push(slot);
        }
    }
    Ok(configured)
}

/// `cluster` shows the accounts the next cluster uses. `use` and `reset` change them.
fn handle_cluster(sub: ClusterSubcommand) -> Result<()> {
    let mut settings = load_settings()?;
    match sub {
        ClusterSubcommand::Show => {
            let configured = configured_slots()?;
            if configured.is_empty() {
                println!("No accounts are configured yet. Run `compute-pool login`.");
                return Ok(());
            }
            let chosen = cluster_slots()?;
            println!("\n{}", "Cluster accounts".bold());
            for slot in &configured {
                let user = load_credentials(*slot)?.map(|c| c.username).unwrap_or_default();
                let mark = if chosen.contains(slot) {
                    "in cluster".green().to_string()
                } else {
                    "not used".dimmed().to_string()
                };
                println!("  Slot {:<3} {:<28} {}", slot, user, mark);
            }
            let how = if settings.cluster_slots.is_some() {
                "chosen with `cluster use`"
            } else {
                "all configured accounts, the default"
            };
            println!("\n  The cluster uses {} accounts ({}): {}.", chosen.len(), how, join_slots(&chosen));
            println!("  Change it with `cluster use <slots>`, e.g. `cluster use 1 3`. Go back to all with `cluster reset`.");
        }
        ClusterSubcommand::Use { slots } => {
            let chosen = compute_pool_core::auth::resolve_cluster_slots(&configured_slots()?, Some(&slots))?;
            if chosen.len() < 2 {
                anyhow::bail!("A cluster needs at least 2 accounts. Choose more with `cluster use <slots>`.");
            }
            settings.cluster_slots = Some(chosen.clone());
            save_settings(&settings)?;
            println!("{} The cluster will use accounts {}.", "*".green().bold(), join_slots(&chosen));
        }
        ClusterSubcommand::Reset => {
            settings.cluster_slots = None;
            save_settings(&settings)?;
            println!("{} The cluster will use every configured account again.", "*".green().bold());
        }
    }
    Ok(())
}

/// Set the web terminal / file manager password. If one is already set, the
/// current password is required first. The previous password is shown once,
/// so the user knows what still works for a session already running. The new
/// password is never printed: it takes effect at the next `compute-pool shell`.
fn set_web_password() -> Result<()> {
    let mut settings = load_settings()?;
    let previous = settings.web_password.clone();

    if let Some(current) = &previous {
        let entered = prompt_masked_password("Current web password")?;
        if &entered != current {
            anyhow::bail!("Current password is wrong. Nothing was changed.");
        }
    }

    let new = prompt_masked_password("New web password (letters, digits, and - _ . ! @ # % + =)")?;
    validate_web_password(&new)?;
    let again = prompt_masked_password("Repeat the new password")?;
    if new != again {
        anyhow::bail!("The two new passwords do not match. Nothing was changed.");
    }

    settings.web_password = Some(new);
    save_settings(&settings)?;

    match previous {
        Some(old) => {
            println!("{} Previous password: {}", "*".green().bold(), old.bold());
            println!("  It stays in effect for a session that is already running.");
            println!("  The new password is saved and takes effect at the next `compute-pool shell`.");
        }
        None => println!(
            "{} Web password saved. It takes effect at the next `compute-pool shell`.",
            "*".green().bold()
        ),
    }
    Ok(())
}

/// The lowest slot with no account stored, so a first login never asks for a slot.
fn next_free_slot() -> Result<usize> {
    for slot in 1..=MAX_ACCOUNT_SLOTS {
        if load_credentials(slot)?.is_none() {
            return Ok(slot);
        }
    }
    anyhow::bail!("All {} slots are configured. Pass --slot N to replace one.", MAX_ACCOUNT_SLOTS);
}

async fn handle_login(slot: Option<usize>, username: Option<String>, key: Option<String>) -> Result<()> {
    let slot = match slot {
        Some(s) => s,
        None => next_free_slot()?,
    };
    if slot == 0 || slot > MAX_ACCOUNT_SLOTS {
        anyhow::bail!("Slot must be between 1 and {}.", MAX_ACCOUNT_SLOTS);
    }
    if let Some(existing) = load_credentials(slot)? {
        println!("{} Slot {} already has account {}; it will be replaced.", "!".yellow().bold(), slot, existing.username);
    }
    ensure_policy_accepted()?;

    let u = match username {
        Some(val) => val,
        None => dialoguer::Input::<String>::new()
            .with_prompt(format!("Kaggle username for Slot {}", slot))
            .interact_text()?,
    };

    let k = match key {
        Some(val) => val,
        None => prompt_masked_password(&format!("Kaggle API key / token for Slot {}", slot))?,
    };

    save_credentials(slot, &Credentials { username: u.clone(), key: k })?;
    let cred_path = compute_pool_core::auth::get_credentials_path()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "~/.compute_pool/credentials.json".to_string());

    println!(
        "{} Saved credentials for Slot {} ({})",
        "*".green().bold(),
        slot,
        u.bold()
    );
    println!("   {} Stored at: {}", "->".cyan(), cred_path.cyan());
    Ok(())
}

async fn handle_accounts_list() -> Result<()> {
    println!("\n{}", "Compute Pool -- Configured Accounts".bold());
    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_content_arrangement(ContentArrangement::Dynamic);
    table.set_header(vec!["Slot", "Username", "Status"]);

    for slot in 1..=MAX_ACCOUNT_SLOTS {
        match load_credentials(slot)? {
            Some(creds) => {
                table.add_row(vec![
                    Cell::new(slot.to_string()),
                    Cell::new(creds.username),
                    Cell::new("Configured").fg(Color::Green),
                ]);
            }
            None => {
                table.add_row(vec![
                    Cell::new(slot.to_string()),
                    Cell::new("<not configured>").fg(Color::DarkGrey),
                    Cell::new("Missing").fg(Color::Red),
                ]);
            }
        }
    }

    println!("{table}\n");
    Ok(())
}

async fn handle_accounts_status() -> Result<()> {
    println!("\n{}", "Compute Pool -- Live Account Quotas".bold());
    println!("{}", "Fetching real-time quota status from Kaggle...".dimmed());

    let statuses = get_all_statuses().await;

    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_content_arrangement(ContentArrangement::Dynamic);
    table.set_header(vec![
        "Slot",
        "Username",
        "Status",
        "GPU Used",
        "GPU Remaining",
        "GPU Total",
    ]);

    for s in &statuses {
        let (status_cell, used_cell, rem_cell, tot_cell) = if s.connected {
            (
                Cell::new("Connected").fg(Color::Green),
                Cell::new(format!("{:.2}h", s.gpu_hours_used())),
                Cell::new(format!("{:.2}h", s.gpu_hours_remaining())).fg(Color::Cyan),
                Cell::new(format!("{:.1}h", s.gpu_hours_total())),
            )
        } else {
            (
                Cell::new(s.error.clone().unwrap_or_else(|| "Error".to_string())).fg(Color::Red),
                Cell::new("-"),
                Cell::new("-"),
                Cell::new("-"),
            )
        };

        table.add_row(vec![
            Cell::new(s.slot.to_string()),
            Cell::new(&s.username),
            status_cell,
            used_cell,
            rem_cell,
            tot_cell,
        ]);
    }

    let connected_slots: Vec<_> = statuses.iter().filter(|s| s.connected).collect();
    if !statuses.is_empty() {
        // Every node runs at the same time, so a cluster hour uses one hour from
        // each account. The cluster's budget is the account with the least time left.
        if connected_slots.len() == statuses.len() {
            let wall_used = statuses.iter().map(|s| s.gpu_hours_used()).fold(0.0, f64::max);
            let wall_total = statuses.iter().map(|s| s.gpu_hours_total()).fold(f64::INFINITY, f64::min);
            let wall_remaining = statuses.iter().map(|s| s.gpu_hours_remaining()).fold(f64::INFINITY, f64::min);

            table.add_row(vec![
                Cell::new("Cluster").fg(Color::Yellow),
                Cell::new(format!("{} Nodes, run together", statuses.len())),
                Cell::new("Connected").fg(Color::Green),
                Cell::new(format!("{:.2}h", wall_used)),
                Cell::new(format!("{:.2}h wall-clock", wall_remaining)).fg(Color::Cyan),
                Cell::new(format!("{:.1}h wall-clock", wall_total)),
            ]);
        } else {
            table.add_row(vec![
                Cell::new("Cluster").fg(Color::Yellow),
                Cell::new(format!("{} of {} Nodes connected", connected_slots.len(), statuses.len())),
                Cell::new("Unavailable").fg(Color::Red),
                Cell::new("-"),
                Cell::new("0.00h (a node is disconnected)").fg(Color::Red),
                Cell::new("-"),
            ]);
        }
    }

    println!("{table}\n");
    Ok(())
}

async fn handle_probe(slot: usize) -> Result<()> {
    println!("\n{}", format!("Probing GPU hardware for Slot {}...", slot).bold().cyan());
    println!("{}", "Submitting remote probe kernel to Kaggle cluster...".dimmed());

    let info = run_probe(slot).await?;
    println!("\n{}", format!("GPU Hardware Report -- Slot {} ({})", slot, info.username).bold());
    println!("  GPU Model:      {}", info.gpu_name.green().bold());
    println!("  GPU Count:      {}", info.gpu_count);
    println!("  Total VRAM:     {} GB ({} MiB)", info.vram_gb(), info.vram_mb);
    println!("  Driver Version: {}", info.driver_version);
    println!("  CUDA Version:   {}", info.cuda_version);
    if let Some(err) = info.error {
        println!("  Notice:         {}", err.yellow());
    }
    println!();
    Ok(())
}

async fn handle_jobs_list() -> Result<()> {
    let mut jobs = load_all_jobs()?;
    println!("\n{}", "Compute Pool -- Jobs".bold());

    if jobs.is_empty() {
        println!("{}", "No jobs submitted yet.".dimmed());
        return Ok(());
    }

    // Auto-refresh active jobs against Kaggle
    for job in jobs.iter_mut() {
        if job.state.is_active() {
            if let (Some(slot_num), Some(slug)) = (job.get_slot_number(), &job.kaggle_kernel_slug) {
                if let Ok(Some(creds)) = load_credentials(slot_num) {
                    let client = KaggleClient::new(&creds.username, &creds.key);
                    let kernel_ref = format!("{}/{}", creds.username, slug);
                    if let Ok(st) = client.get_kernel_status(&kernel_ref).await {
                        if st.contains("COMPLETE") {
                            let out_dir = PathBuf::from("data").join("jobs").join(&job.id);
                            let _ = client.download_kernel_output(&kernel_ref, &out_dir).await;
                            job.transition(JobState::Completed, None);
                            let _ = upsert_job(job);
                        } else if st.contains("ERROR") || st.contains("FAILED") {
                            job.transition(JobState::Failed, Some(st));
                            let _ = upsert_job(job);
                        } else if st.contains("CANCEL") {
                            job.transition(JobState::Cancelled, None);
                            let _ = upsert_job(job);
                        } else if st.contains("RUNNING") && job.state != JobState::Running {
                            job.transition(JobState::Running, None);
                            let _ = upsert_job(job);
                        }
                    }
                }
            }
        }
    }

    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_content_arrangement(ContentArrangement::Dynamic);
    table.set_header(vec![
        "Job ID",
        "Name",
        "State",
        "Slot",
        "Account",
        "Submitted",
    ]);

    for job in &jobs {
        let state_cell = match job.state {
            JobState::Completed => Cell::new(job.state.as_str()).fg(Color::Green),
            JobState::Running => Cell::new(job.state.as_str()).fg(Color::Cyan),
            JobState::Failed => Cell::new(job.state.as_str()).fg(Color::Red),
            JobState::Cancelled => Cell::new(job.state.as_str()).fg(Color::Yellow),
            _ => Cell::new(job.state.as_str()).fg(Color::White),
        };

        let slot_str = job
            .assigned_slot
            .as_ref()
            .map(|s| s.to_string())
            .unwrap_or_else(|| "-".to_string());
        let user_str = job.assigned_username.clone().unwrap_or_else(|| "-".to_string());
        let sub_str = if job.created_at.len() >= 19 {
            job.created_at[..19].replace('T', " ")
        } else {
            job.created_at.clone()
        };

        table.add_row(vec![
            Cell::new(&job.id).fg(Color::Yellow),
            Cell::new(&job.spec.name),
            state_cell,
            Cell::new(slot_str),
            Cell::new(user_str),
            Cell::new(sub_str),
        ]);
    }

    println!("{table}\n");
    Ok(())
}

async fn handle_jobs_submit(args: SubmitArgs) -> Result<()> {
    let script_code = match (args.script, args.code) {
        (Some(path), _) => std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read script file: {}", path.display()))?,
        (None, Some(code)) => code,
        (None, None) => anyhow::bail!("Must provide either --script <path> or --code <code>"),
    };

    let next_id = next_job_id()?;
    let mut job = Job::new(
        next_id.clone(),
        JobSpec {
            name: args.name,
            script: script_code,
            gpu: args.gpu,
            gpu_memory_gb: if args.gpu { 15.0 } else { 0.0 },
            max_runtime_hours: args.runtime_hours,
            checkpointable: true,
            max_retries: 3,
        },
    );

    println!("\n{} Job {} submitted.", "*".green().bold(), next_id.bold());
    println!("{}", "Scheduling to available GPU slot...".dimmed());

    if let Err(e) = schedule_job(&mut job).await {
        println!("{} Scheduling failed: {}", "x".red().bold(), e);
        return Ok(());
    }

    println!(
        "{} Assigned to Slot {} ({})",
        "*".green().bold(),
        job.assigned_slot.as_ref().map(|s| s.to_string()).unwrap_or_else(|| "-".to_string()),
        job.assigned_username.as_deref().unwrap_or("-")
    );

    // Run remote
    let slot_num: usize = job.assigned_slot.as_ref().and_then(|v| v.as_u64()).unwrap_or(1) as usize;
    let creds = load_credentials(slot_num)?.context("Assigned credentials missing")?;
    let slug = format!("{}-{}", job.spec.name, job.id);
    let client = KaggleClient::new(&creds.username, &creds.key);

    job.kaggle_kernel_slug = Some(slug.clone());
    job.transition(JobState::Running, None);
    upsert_job(&job)?;

    println!(
        "{} Dispatching kernel to Kaggle GPU worker...",
        "*".cyan().bold()
    );
    client
        .push_kernel(&slug, &job.spec.script, job.spec.gpu)
        .await?;

    let kernel_ref = format!("{}/{}", creds.username, slug);
    let start = Instant::now();
    let deadline = Duration::from_secs((job.spec.max_runtime_hours * 3600.0) as u64 + 300);
    let mut dots = 0;
    let mut final_status = "UNKNOWN".to_string();

    while start.elapsed() < deadline {
        if let Ok(st) = client.get_kernel_status(&kernel_ref).await {
            print!(
                "\r  Worker status: {} {}    ",
                st.bold(),
                ".".repeat(dots % 4 + 1)
            );
            dots += 1;
            if st.contains("COMPLETE") {
                println!();
                final_status = "COMPLETE".to_string();
                break;
            } else if st.contains("ERROR") || st.contains("FAILED") {
                println!();
                final_status = "ERROR".to_string();
                break;
            } else if st.contains("CANCEL") {
                println!();
                final_status = "CANCELLED".to_string();
                break;
            }
        }
        tokio::time::sleep(Duration::from_secs(8)).await;
    }

    let out_dir = PathBuf::from("data").join("jobs").join(&job.id);
    let _ = client.download_kernel_output(&kernel_ref, &out_dir).await;

    if final_status == "COMPLETE" {
        job.transition(JobState::Completed, None);
        println!(
            "\n{} Job {} completed successfully on GPU!\n",
            "*".green().bold(),
            job.id
        );
    } else {
        job.transition(
            JobState::Failed,
            Some(format!("Execution ended with status: {}", final_status)),
        );
        println!(
            "\n{} Job {} failed ({})\n",
            "x".red().bold(),
            job.id,
            final_status
        );
    }

    upsert_job(&job)?;
    Ok(())
}

async fn handle_jobs_status(job_id: &str) -> Result<()> {
    let mut job_opt = get_job(job_id)?;

    // Auto-refresh against Kaggle if active
    if let Some(ref mut job) = job_opt {
        if job.state.is_active() {
            if let (Some(slot_num), Some(slug)) = (job.get_slot_number(), &job.kaggle_kernel_slug) {
                if let Ok(Some(creds)) = load_credentials(slot_num) {
                    let client = KaggleClient::new(&creds.username, &creds.key);
                    let kernel_ref = format!("{}/{}", creds.username, slug);
                    if let Ok(st) = client.get_kernel_status(&kernel_ref).await {
                        if st.contains("COMPLETE") {
                            let out_dir = PathBuf::from("data").join("jobs").join(&job.id);
                            let _ = client.download_kernel_output(&kernel_ref, &out_dir).await;
                            job.transition(JobState::Completed, None);
                            let _ = upsert_job(job);
                        } else if st.contains("ERROR") || st.contains("FAILED") {
                            job.transition(JobState::Failed, Some(st));
                            let _ = upsert_job(job);
                        } else if st.contains("CANCEL") {
                            job.transition(JobState::Cancelled, None);
                            let _ = upsert_job(job);
                        } else if st.contains("RUNNING") && job.state != JobState::Running {
                            job.transition(JobState::Running, None);
                            let _ = upsert_job(job);
                        }
                    }
                }
            }
        }
    }

    match job_opt {
        Some(job) => {
            println!("\n{}", format!("Job Details: {}", job.id).bold());
            println!("  Name:         {}", job.spec.name);
            println!("  State:        {}", job.state.as_str());
            println!(
                "  Slot:         {}",
                job.assigned_slot
                    .as_ref()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "-".to_string())
            );
            println!(
                "  Account:      {}",
                job.assigned_username.unwrap_or_else(|| "-".to_string())
            );
            println!(
                "  Kernel Slug:  {}",
                job.kaggle_kernel_slug.unwrap_or_else(|| "-".to_string())
            );
            println!("  Created:      {}", job.created_at);
            println!("  Updated:      {}", job.updated_at);
            if let Some(err) = job.error {
                println!("  Error:        {}", err.red());
            }
            println!();
        }
        None => {
            println!("{} Job {} not found.", "x".red().bold(), job_id);
        }
    }
    Ok(())
}

async fn handle_jobs_logs(job_id: &str) -> Result<()> {
    let job = get_job(job_id)?
        .ok_or_else(|| anyhow::anyhow!("Job {} not found", job_id))?;

    let out_dir = PathBuf::from("data").join("jobs").join(&job.id);
    if !out_dir.exists() {
        println!("{} No local logs found for Job {}.", "x".yellow(), job.id);
        return Ok(());
    }

    let mut found = false;
    if let Ok(entries) = std::fs::read_dir(&out_dir) {
        for entry in entries.flatten() {
            if entry.path().extension().map(|e| e == "log").unwrap_or(false) {
                let content = std::fs::read_to_string(entry.path())?;
                println!("\n{}", format!("--- Execution Logs ({}) ---", job.id).bold());
                for line in content.lines() {
                    println!("  {}", line);
                }
                println!("{}\n", "--------------------------------------".bold());
                found = true;
            }
        }
    }

    if !found {
        println!("{} No log files in {}", "!".yellow(), out_dir.display());
    }
    Ok(())
}

async fn handle_jobs_stop(job_id: &str) -> Result<()> {
    let mut job = get_job(job_id)?
        .ok_or_else(|| anyhow::anyhow!("Job {} not found", job_id))?;

    let slot = job.get_slot_number();
    let slug = job.kaggle_kernel_slug.clone();

    // If it's an interactive shell or cluster, kill the tunnels & processes immediately
    if job.spec.name.contains("shell") || job.spec.name.contains("cluster") {
        let _ = compute_pool_core::shell::stop_gpu_shell(slot).await;
    }

    // If a session_id is saved in the script field, send instant ntfy STOP
    if let Some(session_id) = job.spec.script.strip_prefix("session_id:") {
        let http_client = reqwest::Client::new();
        let _ = http_client.post(&format!("https://ntfy.sh/{}-stop", session_id.trim())).body("STOP").send().await;
    }

    // Push exit script to remote Kaggle kernel to terminate worker
    if let (Some(slot_num), Some(ref s)) = (slot, &slug) {
        if let Ok(Some(creds)) = load_credentials(slot_num) {
            let client = KaggleClient::new(&creds.username, &creds.key);
            let _ = client
                .push_kernel(s, "import sys\nprint('Terminated by user.')\nsys.exit(0)\n", false)
                .await;
        }
    }

    job.transition(JobState::Cancelled, Some("Cancelled by user".to_string()));
    upsert_job(&job)?;
    println!("{} Job {} stopped and remote GPU worker terminated.", "*".green().bold(), job.id);
    Ok(())
}

fn handle_jobs_delete(job_id: &str) -> Result<()> {
    if let Some(_) = delete_job(job_id)? {
        println!("{} Job {} deleted.", "*".green().bold(), job_id);
    } else {
        println!("{} Job {} not found.", "x".red().bold(), job_id);
    }
    Ok(())
}

fn handle_jobs_clear(force: bool) -> Result<()> {
    if !force {
        let confirmed = dialoguer::Confirm::new()
            .with_prompt("Are you sure you want to delete all job records?")
            .default(false)
            .interact()?;
        if !confirmed {
            println!("Aborted.");
            return Ok(());
        }
    }

    let deleted = clear_jobs(true)?;
    println!("{} Cleared {} job records.", "*".green().bold(), deleted.len());
    Ok(())
}

fn handle_jobs_reindex() -> Result<()> {
    let jobs = reindex_jobs()?;
    println!(
        "{} Successfully reindexed {} jobs sequentially (0, 1, 2, ...).",
        "*".green().bold(),
        jobs.len()
    );
    Ok(())
}

async fn handle_shell(
    slot_arg: &str,
    duration: u32,
    open: bool,
    timeout: Option<u64>,
) -> Result<()> {
    ensure_policy_accepted()?;
    if load_settings()?.web_password.is_none() {
        println!(
            "{}",
            "No web password is set yet. Set one now; it protects the web terminal and file manager.".yellow()
        );
        set_web_password()?;
    }
    if slot_arg.eq_ignore_ascii_case("cluster") || slot_arg == "0" {
        let slots = cluster_slots()?;
        let nodes = slots.len();
        println!("\n{}", format!("Connecting {}-node GPU task cluster (accounts {})...", nodes, join_slots(&slots)).bold().cyan());
        println!("{}", format!("  • Checking GPU quota on {} accounts...", nodes).dimmed());
        println!("{}", "  • Submitting master and worker kernels in parallel...".dimmed());
        println!("{}", "  • Bridging each worker over a Chisel/Cloudflare tunnel...".dimmed());

        let info = launch_cluster_shell(slots, duration, timeout).await?;
        println!("\n{}", "Compute Pool -- GPU Task Cluster".bold().green());
        println!("  Nodes:                 {} (one Kaggle account each)", info.nodes.len());
        println!("  Master Web Terminal:   {}", info.web_url.bold().cyan());
        for node in &info.nodes {
            let role = if node.node_index == 0 { "Master" } else { "Worker" };
            println!("  node{} ({}) files:     {}", node.node_index, role, if node.files_url.is_empty() { "-".to_string() } else { node.files_url.bold().cyan().to_string() });
        }
        println!("  Cluster Runner:        crun <command> (e.g. crun nvidia-smi)");
        println!("  Cluster Dispatcher:    crun --gpus <command template>");
        println!("  Duration:              {} mins", info.duration_minutes);
        if open {
            let _ = open::that(&info.web_url);
        }
    } else {
        let slot: usize = slot_arg
            .parse()
            .context("Slot must be a number from 1 to 8, or 'cluster'")?;
        let info = launch_gpu_shell(slot, duration, timeout).await?;
        println!("\n{}", "Compute Pool -- Interactive GPU Terminal".bold().green());
        println!("  Slot: {}", info.slot);
        println!("  Account: {}", info.username);
        println!("  Web Terminal: {}", info.web_url.bold().cyan());
        if !info.files_url.is_empty() {
            println!("  File Manager: {}", info.files_url.bold().cyan());
        }
        println!("  Duration: {} mins", info.duration_minutes);
        if open {
            let _ = open::that(&info.web_url);
        }
    }
    Ok(())
}

async fn handle_shell_stop(slot: Option<usize>) -> Result<()> {
    stop_gpu_shell(slot).await?;
    println!(
        "{} GPU shell sessions terminated and resources released.",
        "*".green().bold()
    );
    Ok(())
}
