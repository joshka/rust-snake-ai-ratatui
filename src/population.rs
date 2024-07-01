//! Population
//! A group of simulation agents

use std::cmp::Ordering;

use color_eyre::Result;
use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;
use rayon::prelude::*;

use crate::{
    agent::Agent,
    config::{
        AGENT_COUNT, LOAD_NET, NN_ARCH, POP_NUM_RANDOM, POP_RETAINED, POP_RETAINED_MUTATED,
        POP_ROULETTE, POP_TOURNAMENT,
    },
    neural_net::NeuralNet,
};

pub struct Population {
    agents: Vec<Agent>,
}

impl Population {
    pub fn new() -> Result<Self> {
        let mut agents = vec![];
        for _ in 0..AGENT_COUNT {
            let value = Agent::new(LOAD_NET)?;
            agents.push(value);
        }

        Ok(Self { agents })
    }

    pub fn update(&mut self) -> usize {
        let agents_dead = self
            .agents
            .par_iter_mut()
            .map(|a| {
                a.update().unwrap();
                if a.game.is_dead {
                    1
                } else {
                    0
                }
            })
            .sum::<usize>();

        AGENT_COUNT - agents_dead
    }

    pub fn get_gen_summary(&self) -> Result<(NeuralNet, usize)> {
        let mut max_score = 0;
        let mut best_net = None;

        for a in self.agents.iter() {
            let score = a.game.score();
            if score > max_score {
                max_score = score;
                best_net = Some(a.brain.clone());
            }
        }

        if let Some(net) = best_net {
            return Ok((net, max_score));
        }

        let net = NeuralNet::new(&NN_ARCH)?;
        Ok((net, max_score))
    }

    pub fn reset(&mut self) -> Result<()> {
        let mut new_agents = Vec::with_capacity(AGENT_COUNT);

        // Sort agents based on their fitness
        let mut agents_sorted = self.agents.clone();
        agents_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));

        // Population Distribution
        let num_elite = (AGENT_COUNT as f32 * POP_RETAINED) as usize;
        let num_roulette = (AGENT_COUNT as f32 * POP_ROULETTE) as usize;
        let mut num_tournament = (AGENT_COUNT as f32 * POP_TOURNAMENT) as usize;
        let num_mutated = (AGENT_COUNT as f32 * POP_RETAINED_MUTATED) as usize;
        let num_random = (AGENT_COUNT as f32 * POP_NUM_RANDOM) as usize;

        // Elitism
        // Preserve best performing agents
        // Hels maintain high fitness levels within the population
        for agent in agents_sorted.iter().take(num_elite) {
            let old_brain = agent.brain.clone();
            let new_agent = Agent::with_brain(old_brain);
            new_agents.push(new_agent);
        }

        // Roulette Selection (or Fitness Proportionate Selection)
        // Each agent is selected with a probability proportional to its fitness
        let gene_pool = self.generate_gene_pool();
        if let Some(pool) = gene_pool {
            let mut rng = rand::thread_rng();
            for _ in 0..num_roulette as i32 {
                let rand_parent_1 = self.agents[pool.sample(&mut rng)].clone();
                let rand_parent_2 = self.agents[pool.sample(&mut rng)].clone();
                let mut new_brain = rand_parent_1.brain.merge(&rand_parent_2.brain)?;
                new_brain.mutate();

                let new_agent = Agent::with_brain(new_brain);
                new_agents.push(new_agent);
            }
        } else {
            num_tournament += num_roulette;
        }

        // Tournament Selection
        // Fittest agents among a randomly selected group (tournament)
        // Tournament Size (TS) controls the balance between exploration and exploitation
        // Smaller TS -> More exploration
        let tournament_size = 5;
        for _ in 0..num_tournament {
            let winner = self.tournament_selection(tournament_size);
            let mut new_brain = winner.brain.clone();
            new_brain.mutate();
            new_agents.push(Agent::with_brain(new_brain));
        }

        // Mutational Elitism
        // Allows for incremental improvements to already good solutions
        for agent in agents_sorted.iter().take(num_mutated) {
            let mut old_brain = agent.brain.clone();
            old_brain.mutate();
            new_agents.push(Agent::with_brain(old_brain));
        }

        // Full random
        // Diversify the gene pool
        for _ in 0..num_random as i32 {
            new_agents.push(Agent::new(false)?);
        }

        self.agents = new_agents;
        Ok(())
    }

    fn tournament_selection(&self, tournament_size: usize) -> &Agent {
        let mut rng = rand::thread_rng();
        let mut best_agent = &self.agents[rng.gen_range(0..self.agents.len())];

        for _ in 0..tournament_size {
            let agent = &self.agents[rng.gen_range(0..self.agents.len())];
            if agent.fitness() > best_agent.fitness() {
                best_agent = agent;
            }
        }

        best_agent
    }

    fn generate_gene_pool(&self) -> Option<WeightedIndex<f32>> {
        let mut max_fitness = 0.0;
        let mut weights = vec![];

        for a in self.agents.iter() {
            let fitness = a.fitness();
            if fitness > max_fitness {
                max_fitness = fitness;
            }

            if fitness.is_finite() {
                weights.push(fitness);
            }
        }
        weights
            .iter_mut()
            .for_each(|i| *i = (*i / max_fitness) * 100.0);

        WeightedIndex::new(&weights).ok()
    }
}
