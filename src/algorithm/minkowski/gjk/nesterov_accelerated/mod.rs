use std::ops::Neg;

use cgmath::{BaseFloat, InnerSpace, Zero, Array, Transform, vec3, Vector3, Point3};

use crate::{algorithm::minkowski::SupportPoint, Primitive};

use super::{simplex::Simplex, GJK3};


struct NesterovData<S>
where
    S: BaseFloat,
{
    alpha: S,
    omega: S,

    simplex: Simplex<Point3<S>>,
    ray: Vector3<S>,
    ray_len: S,
    ray_dir: Vector3<S>,

    support_point: SupportPoint<Point3<S>>,
}


impl<S> GJK3<S>
where
    S: BaseFloat,
{
    /// TODO
    pub fn intersect_nesterov_accelerated<PL, PR, TL, TR>(
        &self,
        left: &PL,
        left_transform: &TL,
        right: &PR,
        right_transform: &TR,
    ) -> Option<Simplex<Point3<S>>>
    where
        PL: Primitive<Point = Point3<S>>,
        PR: Primitive<Point = Point3<S>>,
        TL: Transform<Point3<S>>,
        TR: Transform<Point3<S>>, 
    {
        let upper_bound = S::from(1000000000).unwrap();

        let mut use_nesterov_acceleration = true;
        let normalize_support_direction = false;

        let inflation = S::zero();

        let mut inside = false;
        let mut distance = S::zero();

        let mut data = NesterovData::new(None, self.distance_tolerance);

        for i in 0..self.max_iterations {
            let k = i as f64;

            if data.ray_len < self.distance_tolerance {
                distance = -inflation;
                inside = true;
                break
            }

            if use_nesterov_acceleration {
                let momentum = S::from((k + 1.0) / (k + 3.0)).unwrap();
                let y = data.ray * momentum + data.support_point.v * (S::one() - momentum);
                data.ray_dir =  data.ray_dir * momentum + y * (S::one() - momentum);

                if normalize_support_direction {
                    data.ray_dir = data.ray_dir.normalize();
                }
            } else {
                data.ray_dir = data.ray;
            }

            data.support_point = SupportPoint::<Point3<S>>::from_minkowski(left, left_transform, right, right_transform, &data.ray_dir.neg());
            data.simplex.push(data.support_point);

            data.omega = data.ray_dir.dot(data.support_point.v) / data.ray_dir.magnitude();
            if data.omega > upper_bound {
                distance = data.omega - inflation;
                inside = false;
                break
            }

            if use_nesterov_acceleration {
                let frank_wolfe_duality_gap = S::from(2.0).unwrap() * data.ray.dot(data.ray - data.support_point.v);
                if frank_wolfe_duality_gap - self.distance_tolerance <= S::zero() {
                    use_nesterov_acceleration = false;
                    data.simplex.pop();
                    continue
                }
            }

            let cv_check_passed = self.check_convergence(&mut data);
            if i > 0 && cv_check_passed {
                data.simplex.pop();
                if use_nesterov_acceleration{
                    use_nesterov_acceleration = false;
                    continue
                }
                distance = data.ray_len - inflation;

                if distance < self.distance_tolerance{
                    inside = true
                }
                break
            }

            match data.simplex.len() {
                1 => { data.ray = data.support_point.v; }
                2 => { inside = self.project_line_origen(&mut data) }
                3 => { inside = self.project_triangle_origen(&mut data) }
                4 => { inside = self.project_tetra_to_origen(&mut data) }
                _ => {}
            }

            if !inside{
                data.ray_len = data.ray.magnitude();
            }

            if inside || data.ray_len == S::zero() {
                distance = -inflation;
                inside = true;
                break
            }
        }

        print!("{:?}", distance);

        return if inside { Some(data.simplex) } else { None };
    }

    fn check_convergence(&self, data: &mut NesterovData<S>) -> bool 
    {
        data.alpha = data.alpha.max(data.omega);

        let diff = data.ray_len - data.alpha;

        return (diff - self.distance_tolerance * data.ray_len) <= S::zero();
    }

    fn origen_to_point(
        &self, 
        data: &mut NesterovData<S>, 
        a_index: usize, 
        a: Vector3<S>)
    {
        data.ray = a;
        data.simplex[0] = data.simplex[a_index];
        data.simplex.truncate(1);
    }

    fn origen_to_segment(
        &self, 
        data: &mut NesterovData<S>, 
        a_index: usize, b_index: usize, 
        a: Vector3<S>, b: Vector3<S>, 
        ab: Vector3<S>, ab_dot_a0: S)
    {
        data.ray = (a * ab.dot(b) + b * ab_dot_a0) / ab.magnitude2();
        data.simplex[0] = data.simplex[b_index];
        data.simplex[1] = data.simplex[a_index];
        data.simplex.truncate(2);
    }

    fn origen_to_triangle(
        &self, 
        data: &mut NesterovData<S>, 
        a_index: usize, b_index: usize, c_index: usize,
        abc: Vector3<S>, abc_dot_a0: S) -> bool
    {
        if abc_dot_a0 == S::zero() {
            data.simplex[0] = data.simplex[c_index];
            data.simplex[1] = data.simplex[b_index];
            data.simplex[3] = data.simplex[a_index];
            data.simplex.truncate(3);

            data.ray = Vector3::from_value(S::zero());
            return true;
        }

        if abc_dot_a0 > S::zero() {
            data.simplex[0] = data.simplex[c_index];
            data.simplex[1] = data.simplex[b_index];
        }
        else {
            data.simplex[0] = data.simplex[b_index];
            data.simplex[1] = data.simplex[c_index];
        }

        data.simplex[3] = data.simplex[a_index];
        data.simplex.truncate(3);

        data.ray = abc * -abc_dot_a0 / abc.magnitude2();
        return false;
    }

    fn project_line_origen(&self, data: &mut NesterovData<S>) -> bool
    {
        let a_index = 1;
        let b_index = 0;

        let a = data.simplex[a_index].v;
        let b = data.simplex[b_index].v;

        let ab = b - a;
        let d = ab.dot(-a);

        if d == S::zero() {
            /* Two extremely unlikely cases:
               - AB is orthogonal to A: should never happen because it means the support
                 function did not do any progress and GJK should have stopped.
               - A == origin
              In any case, A is the closest to the origin */
            self.origen_to_point(data, a_index, a);
            return a.is_zero();
        }

        if d < S::zero() {
            self.origen_to_point(data, a_index, a);
        }
        else {
            self.origen_to_segment(data, a_index, b_index, a, b, ab, d);
        }

        return false;
    }

    fn project_triangle_origen(&self, data: &mut NesterovData<S>) -> bool
    {
        let a_index = 2;
        let b_index = 1;
        let c_index = 0;

        let a = data.simplex[a_index].v;
        let b = data.simplex[b_index].v;
        let c = data.simplex[c_index].v;

        let ab = b - a;
        let ac = c - a;
        let abc = ab.cross(ac);

        let edge_ac2o = abc.cross(ac).dot(-a);

        let t_b = |data: &mut NesterovData<S>| {
            let towards_b = ab.dot(-a);
            if towards_b < S::zero(){
                self.origen_to_point(data, a_index, a);
            }
            else{
                self.origen_to_segment(data, a_index, b_index, a, b, ab, towards_b)
            }
        };

        if edge_ac2o >= S::zero() {
            let towards_c = ac.dot(-a);
            if towards_c >= S::zero() {
                self.origen_to_segment(data, a_index, b_index, a, b, ab, towards_c)
            }
            else{
                t_b(data);
            }
        }
        else {
            let edge_ab2o = ab.cross(abc).dot(-a);
            if edge_ab2o >= S::zero(){
                t_b(data);
            }
            else{
                return self.origen_to_triangle(data, a_index, b_index, c_index, abc, abc.dot(-a))
            }
        }

        return false;
    }

    fn project_tetra_to_origen(&self, data: &mut NesterovData<S>) -> bool
    {
        let a_index = 3;
        let b_index = 2;
        let c_index = 1;
        let d_index = 0;

        let a = data.simplex[a_index].v;
        let b = data.simplex[b_index].v;
        let c = data.simplex[c_index].v;
        let d = data.simplex[d_index].v;

        let aa = a.magnitude2();

        let da = d.dot(a);
        let db = d.dot(b);
        let dc = d.dot(c);
        let dd = d.dot(d);
        let da_aa = da - aa;

        let ca = c.dot(a);
        let cb = c.dot(b);
        let cc = c.dot(c);
        let cd = dc;
        let ca_aa = ca - aa;

        let ba = b.dot(a);
        let bb = b.dot(b);
        let bc = cb;
        let bd = db;
        let ba_aa = ba - aa;
        let ba_ca = ba - ca;
        let ca_da = ca - da;
        let da_ba = da - ba;

        let a_cross_b = a.cross(b);
        let a_cross_c = a.cross(c);

        let region_inside = |data: &mut NesterovData<S>| {
            data.ray = Vector3::zero();
            true
        };

        let region_abc = |data: &mut NesterovData<S>| {
            self.origen_to_triangle(data, a_index, b_index, c_index, (b - a).cross(c - a), -c.dot(a_cross_b))
        };

        let region_acd = |data: &mut NesterovData<S>| {
            self.origen_to_triangle(data, a_index, c_index, d_index, (c - a).cross(d - a), -d.dot(a_cross_c))
        };

        let region_adb = |data: &mut NesterovData<S>| {
            self.origen_to_triangle(data, a_index, d_index, b_index, (d - a).cross(b - a), d.dot(a_cross_b))
        };

        let region_ab = |data: &mut NesterovData<S>| {
            self.origen_to_segment(data, a_index, b_index, a, b, b - a, -ba_aa)
        };

        let region_ac = |data: &mut NesterovData<S>| {
            self.origen_to_segment(data, a_index, c_index, a, c, c - a, -ca_aa)
        };

        let region_ad = |data: &mut NesterovData<S>| {
            self.origen_to_segment(data, a_index, d_index, a, d, d - a, -da_aa)
        };

        let region_a = |data: &mut NesterovData<S>| {
            self.origen_to_point(data, a_index, a)
        };

        if ba_aa <= S::zero() {
            if -d.dot(a_cross_b) <= S::zero() {
                if ba * da_ba + bd * ba_aa - bb * da_aa <= S::zero() {
                    if da_aa <= S::zero() {
                        if ba * ba_ca + bb * ca_aa - bc * ba_aa <= S::zero() {
                            region_abc(data);
                        } else {
                            region_ab(data);
                        }
                    } else {
                        if ba * ba_ca + bb * ca_aa - bc * ba_aa <= S::zero() {
                            if ca * ba_ca + cb * ca_aa - cc * ba_aa <= S::zero() {
                                if ca * ca_da + cc * da_aa - cd * ca_aa <= S::zero() {
                                    region_acd(data);
                                } else {
                                    region_ac(data);
                                }
                            } else {
                                region_abc(data);
                            }
                        } else {
                            region_ab(data);
                        }
                    }
                } else {
                    if da * da_ba + dd * ba_aa - db * da_aa <= S::zero() {
                        region_adb(data);
                    } else {
                        if ca * ca_da + cc * da_aa - cd * ca_aa <= S::zero() {
                            if da * ca_da + dc * da_aa - dd * ca_aa <= S::zero() {
                                region_ad(data);
                            } else {
                                region_acd(data);
                            }
                        } else {
                            if da * ca_da + dc * da_aa - dd * ca_aa <= S::zero() {
                                region_ad(data);
                            } else {
                                region_ac(data);
                            }
                        }
                    }
                }
            } else {
                if c.dot(a_cross_b) <= S::zero() {
                    if ba * ba_ca + bb * ca_aa - bc * ba_aa <= S::zero() {
                        if ca * ba_ca + cb * ca_aa - cc * ba_aa <= S::zero() {
                            if ca * ca_da + cc * da_aa - cd * ca_aa <= S::zero() {
                                region_acd(data);
                            } else {
                                region_ac(data);
                            }
                        } else {
                            region_abc(data);
                        }
                    } else {
                        region_ad(data);
                    }
                } else {
                    if d.dot(a_cross_c) <= S::zero() {
                        if ca * ca_da + cc * da_aa - cd * ca_aa <= S::zero() {
                            if da * ca_da + dc * da_aa - dd * ca_aa <= S::zero() {
                                region_ad(data);
                            } else {
                                region_acd(data);
                            }
                        } else {
                            if ca_aa <= S::zero() {
                                region_ac(data);
                            } else {
                                region_ad(data);
                            }
                        }
                    } else {
                        region_inside(data);
                    }
                }
            }
        } else {
            if ca_aa <= S::zero() {
                if d.dot(a_cross_c) <= S::zero() {
                    if da_aa <= S::zero() {
                        if ca * ca_da + cc * da_aa - cd * ca_aa <= S::zero() {
                            if da * ca_da + dc * da_aa - dd * ca_aa <= S::zero() {
                                if da * da_ba + dd * ba_aa - db * da_aa <= S::zero() {
                                    region_adb(data);
                                } else {
                                    region_ad(data);
                                }
                            } else {
                                region_acd(data);
                            }
                        } else {
                            if ca * ba_ca + cb * ca_aa - cc * ba_aa <= S::zero() {
                                region_ac(data);
                            } else {
                                region_abc(data);
                            }
                        }
                    } else {
                        if ca * ba_ca + cb * ca_aa - cc * ba_aa <= S::zero() {
                            if ca * ca_da + cc * da_aa - cd * ca_aa <= S::zero() {
                                region_acd(data);
                            } else {
                                region_ac(data);
                            }
                        } else {
                            if c.dot(a_cross_b) <= S::zero() {
                                region_abc(data);
                            } else {
                                region_acd(data);
                            }
                        }
                    }
                } else {
                    if c.dot(a_cross_b) <= S::zero() {
                        if ca * ba_ca + cb * ca_aa - cc * ba_aa <= S::zero() {
                            region_ac(data);
                        } else {
                            region_abc(data);
                        }
                    } else {
                        if -d.dot(a_cross_b) <= S::zero() {
                            if da * da_ba + dd * ba_aa - db * da_aa <= S::zero() {
                                region_adb(data);
                            } else {
                                region_ad(data);
                            }
                        } else {
                            return region_inside(data);
                        }
                    }
                }
            } else {
                if da_aa <= S::zero() {
                    if -d.dot(a_cross_b) <= S::zero() {
                        if da * ca_da + dc * da_aa - dd * ca_aa <= S::zero() {
                            if da * da_ba + dd * ba_aa - db * da_aa <= S::zero() {
                                region_adb(data);
                            } else {
                                region_ad(data);
                            }
                        } else {
                            if d.dot(a_cross_c) <= S::zero() {
                                region_acd(data);
                            } else {
                                region_adb(data);
                            }
                        }
                    } else {
                        if d.dot(a_cross_c) <= S::zero() {
                            if da * ca_da + dc * da_aa - dd * ca_aa <= S::zero() {
                                region_ad(data);
                            } else {
                                region_acd(data);
                            }
                        } else {
                            region_inside(data);
                        }
                    }
                } else {
                    region_a(data);
                } 
            }
        }

        return false;    
    }

}


impl<S> NesterovData<S> 
where 
    S: BaseFloat,
{
    fn new(ray_guess: Option<Vector3<S>>, tolerance: S) -> Self 
    { 
        let simplex = Simplex::new();
        let mut ray = ray_guess.unwrap_or(vec3(S::one(), S::zero(), S::zero()));
        let mut ray_len = ray.magnitude();

        if ray_len < tolerance {
            ray = vec3(S::one(), S::zero(), S::zero());
            ray_len = S::one();
        }

        let ray_dir = ray;
        let support_point = SupportPoint::new();

        Self { 
            alpha: S::zero(),
            omega: S::zero(),
            simplex, 
            ray, 
            ray_len, 
            ray_dir, 
            support_point 
        } 
    }
}



#[cfg(test)]
mod tests {
    use cgmath::{Decomposed, Quaternion, Rad, Rotation3, Vector3};
    use crate::{primitive::{Cuboid, Sphere}, algorithm::minkowski::GJK3};
    
    fn transform_3d(
        x: f32,
        y: f32,
        z: f32,
        angle_z: f32,
    ) -> Decomposed<Vector3<f32>, Quaternion<f32>> {
        Decomposed {
            disp: Vector3::new(x, y, z),
            rot: Quaternion::from_angle_z(Rad(angle_z)),
            scale: 1.,
        }
    }

    #[test]
    fn test_gjk_nesterov_accelerated_exact_3d() {
        let shape = Cuboid::new(1., 1., 1.);
        let t = transform_3d(0., 0., 0., 0.);
        let gjk = GJK3::new();
        let p = gjk.intersect_nesterov_accelerated(&shape, &t, &shape, &t);
        assert!(p.is_some());
    }

    #[test]
    fn test_gjk_nesterov_accelerated_sphere() {
        let shape = Sphere::new(1.);
        let t = transform_3d(0., 0., 0., 0.);
        let gjk = GJK3::new();
        let p = gjk.intersect(&shape, &t, &shape, &t);
        assert!(p.is_some());
    }

    #[test]
    fn test_nesterov_accelerated_vs_original(){
        let iterations = 10000;

        for i in 0..iterations {

            let shape_0 = 

        }
    }
}