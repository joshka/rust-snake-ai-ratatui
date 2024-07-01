//! Simulation
//! Manages the evolution of population over multiple generations

use std::io;
use std::time::Instant;

use color_eyre::Result;

use crate::{
    config::{SAVE_BEST_NET, SAVE_FILE_NAME},
    population::Population,
    viz::Viz,
};

pub struct Simulation {
    generation_count: usize,
    population: Population,
    viz: Viz,
    generation_start_time: Instant,
    max_score: usize,
}

#[derive(Default, Clone, Copy)]
pub struct GenerationSummary {
    pub generation_count: usize,
    pub time_elapsed_secs: f32,
    pub generation_max_score: usize,
    pub simulation_max_score: usize,
}

impl Simulation {
    pub fn start() -> Result<Self> {
        Ok(Self {
            viz: Viz::new()?,
            generation_count: 0,
            population: Population::new()?,
            generation_start_time: Instant::now(),
            max_score: 0,
        })
    }

    pub fn stop(&self) -> io::Result<()> {
        Viz::restore_terminal()
    }

    pub fn update(&mut self) -> Result<()> {
        let games_alive = self.population.update();
        if games_alive == 0 {
            self.finish_generation()?;
            self.start_generation()?;
        }

        self.viz.update()?;
        // self.viz.draw();
        Ok(())
    }

    pub fn draw(&mut self) {
        self.viz.draw();
    }

    pub fn start_generation(&mut self) -> Result<()> {
        self.generation_count += 1;
        self.population.reset()
    }

    pub fn finish_generation(&mut self) -> Result<()> {
        let (best_net, gen_max_score) = self.population.get_gen_summary()?;
        if gen_max_score > self.max_score {
            self.max_score = gen_max_score;
            if SAVE_BEST_NET {
                best_net.save(SAVE_FILE_NAME)?;
            }
            self.viz.update_brain(best_net);
        }

        let stats = GenerationSummary {
            generation_count: self.generation_count,
            time_elapsed_secs: self.generation_start_time.elapsed().as_secs_f32(),
            generation_max_score: gen_max_score,
            simulation_max_score: self.max_score,
        };
        self.viz.update_summary(stats);
        self.generation_start_time = Instant::now();
        Ok(())
    }
}
