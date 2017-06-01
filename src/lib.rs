#[cfg(test)] #[macro_use]
extern crate assert_approx_eq;
extern crate rulinalg;

use rulinalg::matrix::{BaseMatrix, Matrix};
use rulinalg::matrix::decomposition::PartialPivLu;
use rulinalg::vector::Vector;

/// Interpolate multidemnsional function using radial basis functions.
pub struct Interpolation<F> {
    /// Points at which the values of the function to be interpolated are given.
    points: Matrix<f64>,
    /// Weights of the interpolation.
    weights: Vec<f64>,
    /// Radial basis function used for the interpolation.
    phi: F,
    /// Whether normalized radial basis functions are used.
    normalized: bool,
}

/// Calculate the euclidean distance of two points.
macro_rules! distance {
    ($dim: expr, $p1: expr, $p2: expr) => { {
        let mut sum = 0.;
        for i in 0..$dim {
            let diff = $p1[i] - $p2[i];
            sum += diff * diff;
        }
        sum.sqrt()
    } }
}

impl<F> Interpolation<F>
    where F: Fn(f64) -> f64
{
    /// Create a new interpolation using a radial basis function `phi`.
    ///
    /// The `n x dim` matrix `points` contains `n` points of dimension `dim`.
    /// `values` contains the values of the function to be interpolated at the
    /// given points. `normalized` determines whether normalized radial basis
    /// functions should be used.
    pub fn new(points: Matrix<f64>, values: &[f64], phi: F,
               normalized: bool) -> Interpolation<F> {
        let dim = points.cols();
        let n = points.rows();
        assert_eq!(values.len(), n);

        let mut rbf: Matrix<f64> = Matrix::zeros(n, n);
        let mut rhs = Vector::new(vec![0.; n]);

        for i in 0..n {
            let mut sum = 0.;
            for j in 0..n {
                let val = phi(distance!(dim, points.row(i), points.row(j)));
                sum += val;
                rbf[[i, j]] = val;
            }
            rhs[i] = if normalized {
                sum * values[i]
            } else {
                values[i]
            };
        }

        let lu = PartialPivLu::decompose(rbf).expect("Matrix not invertible");
        let weights = lu.solve(rhs).expect("Matrix is singular.");

        Interpolation {
            points: points,
            weights: weights.into(),
            phi: phi,
            normalized: normalized,
        }
    }

    /// The dimension of the function being interpolated.
    pub fn dimension(&self) -> usize {
        self.points.cols()
    }

    /// The number of points being interpolated.
    pub fn number_of_points(&self) -> usize {
        self.points.rows()
    }

    /// Interpolate the function at the given point.
    pub fn interpolate(&self, point: &[f64]) -> f64 {
        let dim = self.dimension();
        let n = self.number_of_points();

        assert_eq!(point.len(), dim);

        let mut sum = 0.;
        let mut sumw = 0.;
        for i in 0..n {
            let val = (self.phi)(distance!(dim, point, self.points.row(i)));
            sumw += self.weights[i] * val;
            sum += val;
        }
        if self.normalized { sumw / sum } else { sumw }
    }
}

/// Find the parameter `r0` for which the RBF `phi` approximates `f` best.
///
/// Only works for one-dimensional functions.
///
/// The heuristic assumes the points are sorted.
pub fn find_best_parameter<F, G, I>(points: &[f64], f: F, phi: G, normalized: bool,
                                    r0s: I) -> (f64, f64)
    where F: Fn(f64) -> f64 + Copy,
          G: Fn(f64, f64) -> f64,
          I: Iterator<Item = f64>,
{
    let n = points.len();
    assert_ne!(n, 0);
    let values: Vec<f64> = points.iter().cloned().map(f).collect();

    let mut best_r0 = 0.;
    let mut best_error = std::f64::INFINITY;

    for r0 in r0s {
        println!("{}", r0);
        let mpoints = Matrix::new(n, 1, points.clone());
        let interp = Interpolation::new(mpoints, &values,
            |r| phi(r, r0), normalized);

        let mut error: f64 = 0.;
        for i in 0..n {
            {
                let x = points[i];
                error = error.max((interp.interpolate(&[x]) - f(x)).abs());
            }
            if n > 1 && i < n - 1 {
                let x = 0.5 * (points[i] + points[i + 1]);
                error = error.max((interp.interpolate(&[x]) - f(x)).abs());
            }
        }
        if error < best_error {
            best_r0 = r0;
            best_error = error;
        }
    }

    assert!(best_error.is_finite());
    (best_r0, best_error)
}

/// Multiquadric radial basis function.
///
/// `r0` should be larger than the typical separation of pionts, but smaller
/// than the feature size of the function being interpolated.
pub fn multiquadric(r: f64, r0: f64) -> f64 {
    (r*r + r0*r0).sqrt()
}

/// Inverse multiquadric radial basis function.
///
/// `r0` should be larger than the typical separation of pionts, but smaller
/// than the feature size of the function being interpolated.
pub fn inverse_multiquadric(r: f64, r0: f64) -> f64 {
    1. / (r*r + r0*r0).sqrt()
}

/// Thin-plate spline radial basis function.
///
/// `r0` should be larger than the typical separation of pionts, but smaller
/// than the feature size of the function being interpolated.
pub fn thin_plate(r: f64, r0: f64) -> f64 {
    if r == 0. {
        return 0.;
    }
    r*r * (r/r0).ln()
}

/// Gaußian radial basis function.
///
/// `r0` should be larger than the typical separation of pionts, but smaller
/// than the feature size of the function being interpolated.
///
/// Can yield high accuracy, but is very sensitive to the choice of `r0`.
pub fn gaussian(r: f64, r0: f64) -> f64 {
    (-0.5 * r*r / (r0*r0)).exp()
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_dimensional() {
        fn f(x: f64) -> f64 {
            x.exp()
        }

        const N: usize = 100;
        const MAX: f64 = 2.;
        let mut points = Vec::with_capacity(N);
        let mut values = Vec::with_capacity(N);
        for i in 0..N {
            let x = MAX * (i as f64) / (N as f64);
            points.push(x);
            values.push(f(x));
        }

        macro_rules! try_rbf {
            ($phi: expr, $norm: expr, $tol: expr) => {
                let mpoints = Matrix::new(N, 1, points.clone());
                let interp = Interpolation::new(mpoints, &values, $phi, $norm);

                for i in 0..N {
                    let x = points[i];
                    assert_approx_eq!(interp.interpolate(&[x]), values[i], $tol);
                }

                for i in 0..(N - 2) {
                    let x = points[i] + 0.5 * MAX / (N as f64);
                    println!("{}", i);
                    assert_approx_eq!(interp.interpolate(&[x]), f(x), $tol);
                }
            }
        }

        try_rbf!(|r| multiquadric(r, 0.3), true, 1e-6);
        try_rbf!(|r| inverse_multiquadric(r, 0.8), true, 1e-7);
        try_rbf!(|r| thin_plate(r, 0.4), true, 0.003);
        try_rbf!(|r| gaussian(r, 0.3), true, 1e-12);

        try_rbf!(|r| multiquadric(r, 1.0), false, 1e-7);
        try_rbf!(|r| inverse_multiquadric(r, 0.8), false, 1e-7);
        try_rbf!(|r| thin_plate(r, 0.001), false, 1e-3);
        try_rbf!(|r| gaussian(r, 0.2), false, 1e-7);
    }
}
