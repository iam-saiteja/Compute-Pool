use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use colored::*;
use comfy_table::{presets::UTF8_FULL, Cell, Color, ContentArrangement, Table};
use compute_pool_core::{
    auth::{load_credentials, save_credentials, Credentials},
    distributed::run_distributed_workload,
    job::{Job, JobSpec, JobState},
    kaggle::KaggleClient,
    probe::run_probe,
    scheduler::{get_all_statuses, schedule_job},
    shell::{launch_cluster_shell, launch_gpu_shell, stop_gpu_shell},
    storage::{
        clear_jobs, delete_job, get_job, load_all_jobs, next_job_id, reindex_jobs, upsert_job,
    },
};
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
#[command(
    name = "compute-pool",
    about = "Compute Pool -- Native High-Performance GPU Cluster & Orchestration Engine",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Save Kaggle credentials for an account slot (1 or 2)
    Login {
        /// Account slot number (1 or 2)
        #[arg(short, long, default_value = "1")]
        slot: usize,

        /// Kaggle username
        #[arg(short, long)]
        username: Option<String>,

        /// Kaggle API key / token
        #[arg(short, long)]
        key: Option<String>,
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
        /// Account slot number (1 or 2)
        #[arg(short, long, default_value = "1")]
        slot: usize,
    },

    /// Launch live interactive GPU web terminal (Single node or Unified 4-GPU Cluster)
    Shell {
        /// Account slot (1, 2) or 'cluster' for 4-GPU unified master-worker session
        #[arg(short, long, default_value = "cluster")]
        slot: String,

        /// Session duration in minutes (max 120)
        #[arg(short, long, default_value = "120")]
        duration: u32,

        /// Automatically open Master Web Terminal in default browser
        #[arg(long)]
        open: bool,

        /// Connection timeout in seconds
        #[arg(long, default_value = "240")]
        timeout: u64,
    },

    /// Stop active interactive GPU terminals and release GPU resources immediately
    #[command(name = "shell-stop")]
    ShellStop {
        /// Account slot to stop (1, 2, or leave empty for all)
        #[arg(short, long)]
        slot: Option<usize>,
    },

    /// Multi-node distributed GPU training coordinator
    Distributed {
        #[command(subcommand)]
        sub: DistributedSubcommand,
    },
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

#[derive(Subcommand, Debug)]
enum DistributedSubcommand {
    /// Launch PyTorch DDP / multi-node script across 4 GPUs
    Run {
        /// Python script path
        script: PathBuf,
        /// Job name
        #[arg(short, long, default_value = "distributed-training")]
        name: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Login { slot, username, key } => {
            handle_login(slot, username, key).await?;
        }
        Commands::Accounts { sub } => match sub {
            AccountsSubcommand::List => handle_accounts_list().await?,
            AccountsSubcommand::Status => handle_accounts_status().await?,
        },
        Commands::Jobs { sub } => match sub {
            JobsSubcommand::List => handle_jobs_list()?,
            JobsSubcommand::Submit(args) => handle_jobs_submit(args).await?,
            JobsSubcommand::Status { job_id } => handle_jobs_status(&job_id)?,
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
        Commands::ShellStop { slot } => {
            handle_shell_stop(slot).await?;
        }
        Commands::Distributed { sub } => match sub {
            DistributedSubcommand::Run { script, name } => {
                handle_distributed_run(script, name).await?;
            }
        },
    }

    Ok(())
}

async fn handle_login(slot: usize, username: Option<String>, key: Option<String>) -> Result<()> {
    if slot != 1 && slot != 2 {
        anyhow::bail!("Only Slot 1 and Slot 2 are supported.");
    }

    let u = match username {
        Some(val) => val,
        None => dialoguer::Input::<String>::new()
            .with_prompt(format!("Kaggle username for Slot {}", slot))
            .interact_text()?,
    };

    let k = match key {
        Some(val) => val,
        None => dialoguer::Password::new()
            .with_prompt(format!("Kaggle API key / token for Slot {}", slot))
            .interact()?,
    };

    save_credentials(slot, &Credentials { username: u.clone(), key: k })?;
    println!(
        "{} Saved credentials for Slot {} ({})",
        "*".green().bold(),
        slot,
        u.bold()
    );
    Ok(())
}

async fn handle_accounts_list() -> Result<()> {
    println!("\n{}", "Compute Pool -- Configured Accounts".bold());
    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_content_arrangement(ContentArrangement::Dynamic);
    table.set_header(vec!["Slot", "Username", "Status"]);

    for slot in 1..=2 {
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

    for s in statuses {
        let (status_cell, used_cell, rem_cell, tot_cell) = if s.connected {
            (
                Cell::new("Connected").fg(Color::Green),
                Cell::new(format!("{:.2}h", s.gpu_hours_used())),
                Cell::new(format!("{:.2}h", s.gpu_hours_remaining())).fg(Color::Cyan),
                Cell::new(format!("{:.1}h", s.gpu_hours_total())),
            )
        } else {
            (
                Cell::new(s.error.unwrap_or_else(|| "Error".to_string())).fg(Color::Red),
                Cell::new("-"),
                Cell::new("-"),
                Cell::new("-"),
            )
        };

        table.add_row(vec![
            Cell::new(s.slot.to_string()),
            Cell::new(s.username),
            status_cell,
            used_cell,
            rem_cell,
            tot_cell,
        ]);
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

fn handle_jobs_list() -> Result<()> {
    let jobs = load_all_jobs()?;
    println!("\n{}", "Compute Pool -- Jobs".bold());

    if jobs.is_empty() {
        println!("{}", "No jobs submitted yet.".dimmed());
        return Ok(());
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

    for job in jobs {
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
        let user_str = job.assigned_username.unwrap_or_else(|| "-".to_string());
        let sub_str = if job.created_at.len() >= 19 {
            job.created_at[..19].replace('T', " ")
        } else {
            job.created_at
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
        .push_kernel(&slug, &job.spec.script, job.spec.gpu, Some("nvidia-tesla-t4"))
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

fn handle_jobs_status(job_id: &str) -> Result<()> {
    match get_job(job_id)? {
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

    if let (Some(slot_val), Some(slug)) = (job.assigned_slot.as_ref().and_then(|v| v.as_u64()), &job.kaggle_kernel_slug) {
        let slot = slot_val as usize;
        if let Ok(Some(creds)) = load_credentials(slot) {
            let client = KaggleClient::new(&creds.username, &creds.key);
            let _ = client
                .push_kernel(slug, "import sys\nsys.exit(0)\n", false, None)
                .await;
        }
    }

    job.transition(JobState::Failed, Some("Cancelled by user".to_string()));
    upsert_job(&job)?;
    println!("{} Job {} cancelled.", "*".green().bold(), job.id);
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
    timeout: u64,
) -> Result<()> {
    if slot_arg.eq_ignore_ascii_case("cluster") || slot_arg == "0" {
        let info = launch_cluster_shell(duration, timeout).await?;
        println!("\n{}", "Compute Pool -- Unified 4-GPU Cluster".bold().green());
        println!("  Master Web Terminal: {}", info.web_url.bold().cyan());
        if !info.files_url.is_empty() {
            println!("  Cluster File Manager: {}", info.files_url.bold().cyan());
        }
        println!("  Duration: {} mins", info.duration_minutes);
        if open {
            let _ = open::that(&info.web_url);
        }
    } else {
        let slot: usize = slot_arg
            .parse()
            .context("Slot must be 1, 2, or 'cluster'")?;
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

async fn handle_distributed_run(script: PathBuf, name: String) -> Result<()> {
    let code = std::fs::read_to_string(&script)
        .with_context(|| format!("Failed to read script file: {}", script.display()))?;

    println!("\n{}", "Compute Pool -- Multi-Node Distributed Training".bold().cyan());
    println!("  Script: {}", script.display());
    println!("  Cluster: 2 Nodes (4x Tesla T4 GPUs total)");
    println!("{}", "  Dispatching parallel execution across both GPU slots...".dimmed());

    let res = run_distributed_workload(&name, &code, true, 2.0).await?;

    for node in res.node_results {
        let status_colored = if node.status == "COMPLETE" {
            node.status.green().bold()
        } else {
            node.status.red().bold()
        };
        println!("\n{}", format!("--- Node {} (Slot {}: {}) [{}] ---", node.rank, node.slot, node.username, status_colored).bold());
        for line in node.log.lines().take(40) {
            println!("  {}", line);
        }
    }

    if res.success {
        println!("\n{} Distributed Job {} finished successfully in {:.1}s!\n", "*".green().bold(), res.job_id, res.elapsed_seconds);
    } else {
        println!("\n{} Distributed Job {} failed on one or more nodes.\n", "x".red().bold(), res.job_id);
    }

    Ok(())
}
