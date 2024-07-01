//! A simple Feed-Forward Neural Network
//!
//! It can't do backpropagation
//! It can only be used for neuro-evolution

use std::{
    fs::{create_dir_all, File},
    io,
    iter::{self, zip},
    path::Path,
};

use color_eyre::{
    eyre::{ensure, eyre, WrapErr},
    Result,
};
use itertools::Itertools;
use rand::{rngs::ThreadRng, Rng};
use serde::{Deserialize, Serialize};

use crate::config::{BRAIN_MUTATION_RATE, BRAIN_MUTATION_VARIATION};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNet {
    input_count: usize,
    layers: Vec<Layer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Layer {
    nodes: Vec<Node>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Node {
    weights: Vec<f64>,
    bias: f64,
}

impl NeuralNet {
    /// Creates a new neural network with the given layer sizes
    pub fn new(layer_sizes: &[usize]) -> Result<Self> {
        ensure!(layer_sizes.len() >= 2, "Need at least 2 layers");
        ensure!(
            layer_sizes.iter().all(|&size| size > 0),
            "All layers must have at least 1 node"
        );

        let layers = layer_sizes
            .iter()
            .tuple_windows()
            .map(|(&a, &b)| Layer::new(b, a))
            .collect_vec();

        Ok(Self {
            layers,
            input_count: layer_sizes[0],
        })
    }

    /// Load a neural network from a file
    pub fn load(path: &str) -> Result<Self> {
        let file = File::open(path).wrap_err_with(|| format!("failed to open file: {path}"))?;
        serde_json_path_to_error::from_reader(file).wrap_err("unable to deserialize net")
    }

    /// Save the neural network to a file
    pub fn save(&self, path: &str) -> color_eyre::Result<()> {
        let path = Path::new(path);
        let file = File::create(path).or_else(|err| {
            ensure!(err.kind() == io::ErrorKind::NotFound, err);
            let parent = path
                .parent()
                .ok_or_else(|| eyre!("unable to find parent path for {}", path.display()))?;
            create_dir_all(parent)
                .wrap_err_with(|| format!("unable to create parent path {}", parent.display()))?;
            File::create(path).wrap_err_with(|| format!("unable to create file {}", path.display()))
        })?;

        serde_json_path_to_error::to_writer_pretty(file, &self)
            .wrap_err("failed to write network file")
    }

    /// Merge two neural networks, returning a new one
    pub fn merge(&self, other: &NeuralNet) -> Result<Self> {
        ensure!(
            self.layers.len() == other.layers.len(),
            "unable to merge neural nets with different number of layers"
        );

        let layers = zip(self.layers.iter(), other.layers.iter())
            .map(|(a, b)| a.merge(b))
            .try_collect()?;
        Ok(NeuralNet {
            layers,
            input_count: self.input_count,
        })
    }

    /// Predict the output of the neural network given an input
    pub fn predict(&self, inputs: Vec<f64>) -> Result<Vec<f64>> {
        ensure!(
            inputs.len() == self.input_count,
            "Bad input size, expected {:?} but got {:?}",
            self.input_count,
            inputs.len()
        );
        let outputs = self.layers.iter().fold(inputs, |outputs, layer| {
            layer.predict(&outputs).collect_vec()
        });
        Ok(outputs)
    }

    /// Mutate the neural network
    pub fn mutate(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.mutate()
        }
    }

    // This is for visualization
    pub fn get_bias(&self, layer_idx: usize) -> Vec<f64> {
        self.layers[layer_idx]
            .nodes
            .iter()
            .map(|node| node.bias)
            .collect()
    }
}

impl Layer {
    /// Create a new layer with random weights and biases
    fn new(layer_size: usize, prev_layer_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        let nodes = iter::repeat_with(|| Node {
            weights: iter::repeat_with(|| rng.gen_range(-1.0..1.0))
                .take(prev_layer_size)
                .collect(),
            bias: rng.gen_range(-1.0..1.0),
        })
        .take(layer_size)
        .collect_vec();

        Self { nodes }
    }

    fn merge(&self, other: &Layer) -> Result<Self> {
        ensure!(
            self.nodes.len() == other.nodes.len(),
            "number of nodes must match"
        );
        let mut rng = rand::thread_rng();
        let nodes = zip(self.nodes.iter(), other.nodes.iter())
            .map(|(node1, node2)| node1.merge(node2, &mut rng))
            .collect_vec();
        Ok(Self { nodes })
    }

    fn predict<'a>(&'a self, inputs: &'a [f64]) -> impl Iterator<Item = f64> + '_ {
        self.nodes.iter().map(|node| node.predict(inputs))
    }

    fn mutate(&mut self) {
        let mut rng = rand::thread_rng();

        for node in self.nodes.iter_mut() {
            for val in node.weights.iter_mut() {
                if rng.gen::<f64>() >= BRAIN_MUTATION_RATE {
                    continue;
                }

                *val += rng.gen_range(-BRAIN_MUTATION_VARIATION..BRAIN_MUTATION_VARIATION);
            }
            if rng.gen::<f64>() < BRAIN_MUTATION_RATE {
                node.bias += rng.gen_range(-BRAIN_MUTATION_VARIATION..BRAIN_MUTATION_VARIATION);
            }
        }
    }
}

impl Node {
    fn merge(&self, other: &Node, rng: &mut ThreadRng) -> Self {
        let weights = zip(self.weights.iter(), other.weights.iter())
            .map(|(&w1, &w2)| if rng.gen() { w1 } else { w2 })
            .collect_vec();
        let bias = if rng.gen() { self.bias } else { other.bias };
        Self { weights, bias }
    }

    fn predict(&self, inputs: &[f64]) -> f64 {
        let weighted_sum = self.bias
            + zip(self.weights.iter(), inputs)
                .map(|(weight, value)| weight * value)
                .sum::<f64>();
        // ReLU activation
        weighted_sum.max(0.0)
    }
}
