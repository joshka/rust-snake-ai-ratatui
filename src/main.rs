use std::time::Duration;

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
    loop {
        if event::poll(Duration::from_nanos(1))? {
            if let Event::Key(key) = event::read()? {
                if let KeyCode::Esc | KeyCode::Char('q') = key.code {
                    break;
                }
            }
        }
        simulation.update()?;
    }
    simulation.stop()?;
    Ok(())
}
