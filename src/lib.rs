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

        let lu = PartialPivLu::decompose(rbf).expect("Matrix not invertible.");
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
        const MAX: f64 = 5.;
        let mut points = Vec::with_capacity(N);
        let mut values = Vec::with_capacity(N);
        for i in 0..N {
            let x = MAX / ((N - i) as f64);
            points.push(x);
            values.push(f(x));
        }

        macro_rules! try_rbf {
            ($phi: expr, $norm: expr, $tol_repro: expr, $tol_inter: expr) => {
                let mpoints = Matrix::new(N, 1, points.clone());
                let interp = Interpolation::new(mpoints, &values, $phi, $norm);

                for i in 0..N {
                    let x = points[i];
                    assert_approx_eq!(interp.interpolate(&[x]), values[i], $tol_repro);
                }

                for i in 0..(N - 2) {
                    let x = points[i] + 0.5 * MAX / (N as f64);
                    println!("{}", i);
                    assert_approx_eq!(interp.interpolate(&[x]), f(x), $tol_inter);
                }
            }
        }

        try_rbf!(|r| multiquadric(r, 0.1), false, 1e-6, 1e-2);
        try_rbf!(|r| inverse_multiquadric(r, 0.1), false, 1e-8, 1e-1);
        try_rbf!(|r| thin_plate(r, 0.1), false, 1e-12, 1e-1);
        try_rbf!(|r| gaussian(r, 0.1), false, 1e-7, 3e-1);

        try_rbf!(|r| multiquadric(r, 0.1), true, 1e-5, 1e-1);
        try_rbf!(|r| inverse_multiquadric(r, 0.1), true, 1e-9, 1e-1);
        try_rbf!(|r| thin_plate(r, 0.02), true, 1e-10, 8e-1);
        try_rbf!(|r| gaussian(r, 0.1), true, 1e-11, 2e-1);
    }
}
