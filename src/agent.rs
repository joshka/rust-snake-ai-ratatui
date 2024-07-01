//! AI Agent
//! An instance of the Game played by an AI

use color_eyre::Result;
use itertools::Itertools;

use crate::{
    config::{LOAD_FILE_NAME, NN_ARCH, STEP_COUNT},
    game::Game,
    neural_net::NeuralNet,
    utils::{get_eight_dirs, Direction, Point},
};

#[derive(Clone)]
pub struct Agent {
    pub game: Game,
    pub brain: NeuralNet,
}

impl Agent {
    pub fn new(is_load: bool) -> Result<Self> {
        let brain = if is_load {
            let mut net = NeuralNet::load(LOAD_FILE_NAME)?;
            net.mutate();
            net
        } else {
            NeuralNet::new(&NN_ARCH)?
        };

        Ok(Self {
            game: Game::new(),
            brain,
        })
    }

    pub fn with_brain(brain: NeuralNet) -> Self {
        Self {
            game: Game::new(),
            brain,
        }
    }

    pub fn update(&mut self) -> Result<bool> {
        if self.game.is_dead {
            return Ok(false);
        }

        let direction = self.get_brain_output()?;
        self.game.update(direction);

        // Limit the number of steps the snake can take without eating
        let step_limit = self.get_step_limit();
        if self.game.no_food_steps >= step_limit {
            self.game.is_dead = true;
        }

        Ok(true)
    }

    pub fn fitness(&self) -> f32 {
        let score = self.game.body.len() as f32;
        if score <= 1.0 {
            return 1.0;
        }

        let mut fitness = 1.0;
        if score < 5.0 {
            fitness *= 2.0_f32.powf(score);
            fitness *= score;
            fitness *= self.game.total_steps as f32 * 0.1;
        } else {
            fitness *= score * score;
            fitness *= self.game.total_steps as f32;
        }

        fitness
    }

    pub fn get_brain_output(&self) -> Result<Direction> {
        let vision = self.get_brain_input();
        let cur_dir = self.game.direction;
        let nn_out = self.brain.predict(vision)?.clone();
        let mut max_index = nn_out
            .iter()
            .enumerate()
            .sorted_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i);
        let dir = match max_index.next() {
            Some(0) => Direction::Left,
            Some(1) => Direction::Right,
            Some(2) => Direction::Bottom,
            _ => Direction::Top,
        };

        let direction = if matches!(
            (cur_dir, dir),
            (Direction::Left, Direction::Right)
                | (Direction::Right, Direction::Left)
                | (Direction::Top, Direction::Bottom)
                | (Direction::Bottom, Direction::Top)
        ) {
            // Prevent the snake from turning back on itself by choosing the second highest output
            match max_index.next() {
                Some(0) => Direction::Left,
                Some(1) => Direction::Right,
                Some(2) => Direction::Bottom,
                _ => Direction::Top,
            }
        } else {
            dir
        };

        Ok(direction)
    }

    pub fn get_brain_input(&self) -> Vec<f64> {
        let dirs = get_eight_dirs().to_vec();
        let vision = self.get_snake_vision(dirs);
        let head_dir = self.game.direction.get_one_hot_dir();
        let tail_dir = self.get_tail_direction().get_one_hot_dir();

        vision.into_iter().chain(head_dir).chain(tail_dir).collect()
    }

    fn get_snake_vision(&self, dirs: Vec<(i32, i32)>) -> Vec<f64> {
        let mut vision = vec![];

        for d in dirs {
            // Food and Body are one hot
            let (solid, _food) = self.vision_in_dir(self.game.head, d);
            vision.push(solid as f64);
            vision.push(if _food { 1.0 } else { 0.0 });
        }

        vision
    }

    fn vision_in_dir(&self, st: Point, dir: (i32, i32)) -> (f32, bool) {
        let mut food = false;
        let mut temp_pt: Point = st;
        let mut dist = 0;

        loop {
            if self.game.is_wall(temp_pt) || self.game.is_snake_body(temp_pt) {
                break;
            }

            if self.game.food == temp_pt {
                food = true;
            }

            temp_pt = Point::new(temp_pt.x + dir.0, temp_pt.y + dir.1);

            dist += 1;
            if dist > 1000 {
                break;
            }
        }

        (1.0 / dist as f32, food)
    }

    pub fn get_step_limit(&self) -> usize {
        match self.game.score() {
            score if score > 30 => STEP_COUNT * 6,
            score if score > 20 => STEP_COUNT * 3,
            score if score > 5 => STEP_COUNT * 2,
            _ => STEP_COUNT,
        }
    }

    fn get_tail_direction(&self) -> Direction {
        if let Some(tail) = self.game.body.last() {
            if let Some(body) = self.game.body.get(self.game.body.len() - 2) {
                let x = body.x - tail.x;
                let y = body.y - tail.y;

                return match (x, y) {
                    (-1, 0) => Direction::Left,
                    (1, 0) => Direction::Right,
                    (0, 1) => Direction::Bottom,
                    _ => Direction::Top,
                };
            }
        }

        self.game.direction
    }
}

impl PartialEq for Agent {
    fn eq(&self, other: &Self) -> bool {
        self.fitness() == other.fitness()
    }
}

impl PartialOrd for Agent {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.fitness().partial_cmp(&other.fitness())
    }
}
