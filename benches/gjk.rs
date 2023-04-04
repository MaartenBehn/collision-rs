#![feature(test)]

extern crate test;

use cgmath::{Decomposed, Vector3, Quaternion, BaseFloat, Rotation3, Rad};
use rand::distributions::uniform::{SampleUniform, SampleRange};
use rand::{prelude::StdRng, Rng};
use rand::SeedableRng;
use collision::algorithm::minkowski::GJK3;
use collision::primitive::Primitive3;
use test::{Bencher};

#[bench]
fn test_gjk_original(bench: &mut Bencher) { test_gjk(bench, false); }

#[bench]
fn test_gjk_nasterov(bench: &mut Bencher) { test_gjk(bench, true); }

fn test_gjk(bench: &mut Bencher, use_nasterov: bool) {
    let mut rng = StdRng::seed_from_u64(42);
    let size_range = 0.0..100.0;
    let pos_range = 0.0..500.0;
    let angle_range = 0.0..360.0;

    let gjk = GJK3::new();

    bench.iter(|| {
        let transform_0 = random_transform(&mut rng, pos_range.to_owned(), angle_range.to_owned());
        let transform_1 = random_transform(&mut rng, pos_range.to_owned(), angle_range.to_owned());

        let shape_0 = Primitive3::new_random(&mut rng, size_range.to_owned());
        let shape_1 = Primitive3::new_random(&mut rng, size_range.to_owned());

        if use_nasterov {
            gjk.intersect_nesterov_accelerated(&shape_0, &transform_0, &shape_1, &transform_1);
        } else {
            gjk.intersect(&shape_0, &transform_0, &shape_1, &transform_1);
        }
    });
}

fn random_transform<S, R>(
    rng: &mut impl Rng,
    pos_range: R,
    angle_range: R,
) -> Decomposed<Vector3<S>, Quaternion<S>>
where
    S: BaseFloat + SampleUniform,
    R: SampleRange<S> + Clone,
{
    Decomposed {
        disp: Vector3::<S>::new(
            rng.gen_range(pos_range.to_owned()),
            rng.gen_range(pos_range.to_owned()),
            rng.gen_range(pos_range.to_owned()),
        ),
        rot: Quaternion::from_angle_z(Rad(rng.gen_range(angle_range))),
        scale: S::one(),
    }
}


