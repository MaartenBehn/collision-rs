use cgmath::{Quaternion, Vector3, Decomposed, BaseFloat, Rotation3, Rad, Transform, Matrix4};
use collision::{algorithm::minkowski::GJK3, primitive::{Primitive3, Sphere, Capsule, Cylinder, Cube, Cuboid}};
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use gjk::{json_loder::load_test_file, colliders::{Collider, ColliderType}};
use rand::{rngs::StdRng, Rng, distributions::uniform::{SampleUniform, SampleRange}, SeedableRng};

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

fn original_benchmark(c: &mut Criterion) {

    let mut rng = StdRng::seed_from_u64(42);
    let size_range = 0.0..100.0;
    let pos_range = 0.0..500.0;
    let angle_range = 0.0..360.0;

    let gjk = GJK3::new();

    let mut group = c.benchmark_group("original_gjk");

    for i in 0..10 {
        let transform_0 = random_transform(&mut rng, pos_range.to_owned(), angle_range.to_owned());
        let transform_1 = random_transform(&mut rng, pos_range.to_owned(), angle_range.to_owned());

        let shape_0 = Primitive3::new_random(&mut rng, size_range.to_owned());
        let shape_1 = Primitive3::new_random(&mut rng, size_range.to_owned());


        group.bench_function(BenchmarkId::from_parameter(i), |b| b.iter(|| 
            gjk.intersect(&shape_0, &transform_0, &shape_1, &transform_1)
        ));
    }
    group.finish();
}


fn nasterov_benchmark(c: &mut Criterion) {

    let mut rng = StdRng::seed_from_u64(42);
    let size_range = 0.0..100.0;
    let pos_range = 0.0..500.0;
    let angle_range = 0.0..360.0;

    let gjk = GJK3::new();

    let mut group = c.benchmark_group("nasterov_gjk");

    for i in 0..10 {
        let transform_0 = random_transform(&mut rng, pos_range.to_owned(), angle_range.to_owned());
        let transform_1 = random_transform(&mut rng, pos_range.to_owned(), angle_range.to_owned());

        let shape_0 = Primitive3::new_random(&mut rng, size_range.to_owned());
        let shape_1 = Primitive3::new_random(&mut rng, size_range.to_owned());

        group.bench_function(BenchmarkId::from_parameter(i), |b| b.iter(|| 
            gjk.distance_nesterov_accelerated(&shape_0, &transform_0, &shape_1, &transform_1)
        ));
    }
    group.finish();
}

fn collider_to_transform_and_primitive(collider: &Collider) -> (Matrix4<f64>, Primitive3<f64>){
    let transform = Matrix4::<f64>::new(
        collider.transform.x_axis.x, collider.transform.x_axis.y, collider.transform.x_axis.z, collider.center.x,
        collider.transform.y_axis.x, collider.transform.y_axis.y, collider.transform.y_axis.z, collider.center.y,
        collider.transform.z_axis.x, collider.transform.z_axis.y, collider.transform.z_axis.z, collider.center.z,
        0.0, 0.0, 0.0, 1.0
    );

    let primitive = match collider.typ {
        x if x == ColliderType::Sphere as usize => {
            Primitive3::Sphere(Sphere::new(collider.radius))
        },
        x if x == ColliderType::Capluse as usize => {
            Primitive3::Capsule(Capsule::new(collider.height * 0.5, collider.radius))
        },
        x if x == ColliderType::Cylinder as usize => {
            Primitive3::Cylinder(Cylinder::new(collider.height * 0.5, collider.radius))
        },    
        x if x == ColliderType::Box as usize => {
            Primitive3::Cuboid(Cuboid::new(collider.size.x, collider.size.y, collider.size.z))
        },    
        _ => todo!(),
    };

    (transform, primitive)
}

fn load_data() -> Vec<((Matrix4<f64>, Primitive3<f64>), (Matrix4<f64>, Primitive3<f64>))>{
    let path = "../data/nao_test_cases.json";
    let test_data = load_test_file(path);

    let mut cases = Vec::new();
    for (collider0, collider1, dist) in test_data.iter() {
        cases.push((collider_to_transform_and_primitive(collider0), collider_to_transform_and_primitive(collider1)))
    }

    cases
}

fn original_benchmark_test_file(c: &mut Criterion) {

    let cases = load_data();

    let gjk = GJK3::new();

    c.bench_function("original_intersect_gjk", |b| b.iter(|| 
        for i in 0..cases.len() {
            let ((transform_0, shape_0), (transform_1, shape_1)) = &cases[i];
            gjk.intersect(shape_0, transform_0, shape_1, transform_1);
        }
    ));

    print!("Cases: {:?}", cases.len());
}

fn original_distance_benchmark_test_file(c: &mut Criterion) {

    let cases = load_data();

    let gjk = GJK3::new();

    c.bench_function("original_distance_gjk", |b| b.iter(|| 
        for i in 0..cases.len() {
            let ((transform_0, shape_0), (transform_1, shape_1)) = &cases[i];
            gjk.distance(shape_0, transform_0, shape_1, transform_1);
        }
    ));

    print!("Cases: {:?}", cases.len());
}

fn nasterov_benchmark_test_file(c: &mut Criterion) {

    let cases = load_data();

    let gjk = GJK3::new();

    c.bench_function("nasterov_gjk", |b| b.iter(|| 
        for i in 0..cases.len() {
            let ((transform_0, shape_0), (transform_1, shape_1)) = &cases[i];
            gjk.distance_nesterov_accelerated(shape_0, transform_0, shape_1, transform_1);
        }
    ));

    print!("Cases: {:?}", cases.len());
}

criterion_group!(benches, original_benchmark_test_file, original_distance_benchmark_test_file, nasterov_benchmark_test_file);
criterion_main!(benches);


