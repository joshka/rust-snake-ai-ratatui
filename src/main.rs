use std::time::{Duration, Instant};

use color_eyre::Result;
use ratatui::crossterm::event::{self, Event, KeyCode};

use crate::simulation::Simulation;

mod agent;
mod config;
mod game;
mod neural_net;
mod population;
mod simulation;
mod utils;
mod viz;

fn main() -> Result<()> {
    color_eyre::install()?;
    let mut simulation = Simulation::start()?;
    let mut last_poll = Instant::now();
    loop {
        if last_poll.elapsed() > Duration::from_millis(15) {
            if event::poll(Duration::ZERO)? {
                last_poll = Instant::now();
                if let Event::Key(key) = event::read()? {
                    if let KeyCode::Esc | KeyCode::Char('q') = key.code {
                        break;
                    }
                }
            }
            simulation.draw();
        }
        simulation.update()?;
    }
    simulation.stop()?;
    Ok(())
}
