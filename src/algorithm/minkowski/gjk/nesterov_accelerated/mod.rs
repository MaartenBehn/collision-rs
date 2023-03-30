use std::ops::{Neg, Add};

use cgmath::{EuclideanSpace, BaseFloat, InnerSpace, Zero, Array, UlpsEq, Transform, vec3, Vector3};

use crate::{algorithm::minkowski::{EPA, SupportPoint}, Primitive};

use super::{GJK, SimplexProcessor, simplex::Simplex};


struct NesterovData<S, P>
where
    S: BaseFloat,
    P: EuclideanSpace<Scalar = S>,
{
    alpha: S,
    omega: S,

    simplex: Simplex<P>,
    ray: Vector3<S>,
    ray_len: S,
    ray_dir: Vector3<S>,

    support_point: SupportPoint<P>,
}


impl<SP, E, S> GJK<SP, E, S>
where
    SP: SimplexProcessor,
    SP::Point: EuclideanSpace<Scalar = S>,
    E: EPA<Point = SP::Point>,
    S: BaseFloat,
{
    pub fn intersect_nesterov_accelerated<P, PL, PR, TL, TR>(
        &self,
        left: &PL,
        left_transform: &TL,
        right: &PR,
        right_transform: &TR,
    ) -> Option<Simplex<P>>
    where
        P: EuclideanSpace<Scalar = S>,
        PL: Primitive<Point = P>,
        PR: Primitive<Point = P>,
        SP: SimplexProcessor<Point = P>,
        P::Diff: Neg<Output = P::Diff> + InnerSpace + Zero + Array<Element = S> + UlpsEq,
        TL: Transform<P>,
        TR: Transform<P>, 
    {
        let upper_bound = 1000000000;
        let tolerance = 1e-6;

        let use_nesterov_acceleration = true;
        let normalize_support_direction = false;

        let inflation = 0;

        let mut inside = false;
        let mut distance = 0;

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
                let y = (data.ray * momentum) + (data.support_point.v * (S::one() - momentum));
                data.ray_dir =  data.ray_dir * momentum + y * (S::one() - momentum);

                if normalize_support_direction {
                    data.ray_dir = data.ray_dir.normalize();
                }
            } else {
                data.ray_dir = data.ray;
            }

        }

        return None;
    }

    fn check_convergence<P>(&self, data: &mut NesterovData<S, P>) -> bool 
    where
        P: EuclideanSpace<Scalar = S>
    {
        data.alpha = data.alpha.max(data.omega);

        let diff = data.ray_len - data.alpha;

        return (diff - self.distance_tolerance * data.ray_len) <= S::zero();
    }

    fn origen_to_point<P>(
        &self, 
        data: &mut NesterovData<S, P>, 
        a_index: usize, 
        a: Vector3<S>)
    where
        P: EuclideanSpace<Scalar = S>
    {
        data.ray = a;
        data.simplex[0] = data.simplex[a_index];
        data.simplex.truncate(1);
    }

    fn origen_to_segment<P>(
        &self, 
        data: &mut NesterovData<S, P>, 
        a_index: usize, b_index: usize, 
        a: Vector3<S>, b: Vector3<S>, 
        ab: Vector3<S>, ab_dot_a0: S)
    where
        P: EuclideanSpace<Scalar = S>
    {
        data.ray = (a * ab.dot(b) + b * ab_dot_a0) / ab.magnitude2();
        data.simplex[0] = data.simplex[b_index];
        data.simplex[1] = data.simplex[a_index];
        data.simplex.truncate(2);
    }

    fn origen_to_triangle<P>(
        &self, 
        data: &mut NesterovData<S, P>, 
        a_index: usize, b_index: usize, c_index: usize,
        abc: Vector3<S>, abc_dot_a0: S) -> bool
    where
        P: EuclideanSpace<Scalar = S>
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

    fn project_line_origen<P>(&self, data: &mut NesterovData<S, P>) -> bool
    where
        P: EuclideanSpace<Scalar = S, Diff = Vector3<S>>
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

    fn project_triangle_origen<P>(&self, data: &mut NesterovData<S, P>) -> bool
    where
        P: EuclideanSpace<Scalar = S, Diff = Vector3<S>>
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

        let t_b = || {
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
                t_b();
            }
        }
        else {
            let edge_ab2o = ab.cross(abc).dot(-a);
            if edge_ab2o >= S::zero(){
                t_b();
            }
            else{
                return self.origen_to_triangle(data, a_index, b_index, c_index, abc, abc.dot(-a))
            }
        }

        return false;
    }

    fn project_tetra_to_origen<P>(&self, data: &mut NesterovData<S, P>) -> bool
    where
        P: EuclideanSpace<Scalar = S, Diff = Vector3<S>>
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

        let region_inside = || {
            data.ray = Vector3::zero();
        };

        let region_abc = || {
            self.origen_to_triangle(data, a_index, b_index, c_index, (b - a).cross(c - a), -c.dot(a_cross_b))
        };

        let region_acd = || {
            self.origen_to_triangle(data, a_index, c_index, d_index, (c - a).cross(d - a), -d.dot(a_cross_c))
        };

        let region_adb = || {
            self.origen_to_triangle(data, a_index, d_index, b_index, (d - a).cross(b - a), d.dot(a_cross_b))
        };

        let region_ab = || {
            self.origen_to_segment(data, a_index, b_index, a, b, b - a, -ba_aa)
        };

        let region_ac = || {
            self.origen_to_segment(data, a_index, c_index, a, c, c - a, -ca_aa)
        };

        let region_ad = || {
            self.origen_to_segment(data, a_index, d_index, a, d, d - a, -da_aa)
        };

        let region_a = || {
            self.origen_to_point(data, a_index, a)
        };

        if ba_aa <= S::zero() {
            if -d.dot(a_cross_b) <= S::zero() {
                if ba * da_ba + bd * ba_aa - bb * da_aa <= S::zero() {
                    if da_aa <= S::zero() {
                        if ba * ba_ca + bb * ca_aa - bc * ba_aa <= S::zero() {
                            region_abc();
                        } else {
                            region_ab();
                        }
                    } else {
                        if ba * ba_ca + bb * ca_aa - bc * ba_aa <= S::zero() {
                            if ca * ba_ca + cb * ca_aa - cc * ba_aa <= S::zero() {
                                if ca * ca_da + cc * da_aa - cd * ca_aa <= S::zero() {
                                    region_acd();
                                } else {
                                    region_ac();
                                }
                            } else {
                                region_abc();
                            }
                        } else {
                            region_ab();
                        }
                    }
                } else {
                    if da * da_ba + dd * ba_aa - db * da_aa <= S::zero() {
                        region_adb();
                    } else {
                        if ca * ca_da + cc * da_aa - cd * ca_aa <= S::zero() {
                            if da * ca_da + dc * da_aa - dd * ca_aa <= S::zero() {
                                region_ad();
                            } else {
                                region_acd();
                            }
                        } else {
                            if da * ca_da + dc * da_aa - dd * ca_aa <= S::zero() {
                                region_ad();
                            } else {
                                region_ac();
                            }
                        }
                    }
                }
            } else {
                if c.dot(a_cross_b) <= S::zero() {
                    if ba * ba_ca + bb * ca_aa - bc * ba_aa <= S::zero() {
                        if ca * ba_ca + cb * ca_aa - cc * ba_aa <= S::zero() {
                            if ca * ca_da + cc * da_aa - cd * ca_aa <= S::zero() {
                                region_acd();
                            } else {
                                region_ac();
                            }
                        } else {
                            region_abc();
                        }
                    } else {
                        region_ad();
                    }
                } else {
                    if d.dot(a_cross_c) <= S::zero() {
                        if ca * ca_da + cc * da_aa - cd * ca_aa <= S::zero() {
                            if da * ca_da + dc * da_aa - dd * ca_aa <= S::zero() {
                                region_ad();
                            } else {
                                region_acd();
                            }
                        } else {
                            if ca_aa <= S::zero() {
                                region_ac();
                            } else {
                                region_ad();
                            }
                        }
                    } else {
                        region_inside();
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
                                    region_adb();
                                } else {
                                    region_ad();
                                }
                            } else {
                                region_acd();
                            }
                        } else {
                            if ca * ba_ca + cb * ca_aa - cc * ba_aa <= S::zero() {
                                region_ac();
                            } else {
                                region_abc();
                            }
                        }
                    } else {
                        if ca * ba_ca + cb * ca_aa - cc * ba_aa <= S::zero() {
                            if ca * ca_da + cc * da_aa - cd * ca_aa <= S::zero() {
                                region_acd();
                            } else {
                                region_ac();
                            }
                        } else {
                            if c.dot(a_cross_b) <= S::zero() {
                                region_abc();
                            } else {
                                region_acd();
                            }
                        }
                    }
                } else {
                    if c.dot(a_cross_b) <= S::zero() {
                        if ca * ba_ca + cb * ca_aa - cc * ba_aa <= S::zero() {
                            region_ac();
                        } else {
                            region_abc();
                        }
                    } else {
                        if -d.dot(a_cross_b) <= S::zero() {
                            if da * da_ba + dd * ba_aa - db * da_aa <= S::zero() {
                                region_adb();
                            } else {
                                region_ad();
                            }
                        } else {
                            region_inside();
                        }
                    }
                }
            } else {
                if da_aa <= S::zero() {
                    if -d.dot(a_cross_b) <= S::zero() {
                        if da * ca_da + dc * da_aa - dd * ca_aa <= S::zero() {
                            if da * da_ba + dd * ba_aa - db * da_aa <= S::zero() {
                                region_adb();
                            } else {
                                region_ad();
                            }
                        } else {
                            if d.dot(a_cross_c) <= S::zero() {
                                region_acd();
                            } else {
                                region_adb();
                            }
                        }
                    } else {
                        if d.dot(a_cross_c) <= S::zero() {
                            if da * ca_da + dc * da_aa - dd * ca_aa <= S::zero() {
                                region_ad();
                            } else {
                                region_acd();
                            }
                        } else {
                            region_inside();
                        }
                    }
                } else {
                    region_a();
                } 
            }
        }

        return false;    
    }

}


impl<S, P> NesterovData<S, P> 
where 
    P: EuclideanSpace<Scalar = S>,
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

