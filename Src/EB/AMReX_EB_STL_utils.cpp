#include <AMReX_EB_STL_utils.H>
#include <AMReX_EB_triGeomOps_K.H>
#include <AMReX_IntConv.H>
#include <cstring>

// EY: Timing tool 
#include <chrono>

// EY: CGAL--
// See .H file for more details
#include <iostream>
#include <system_error>
#include <thread>

namespace amrex
{
    //================================================================================
    void STLtools::read_ascii_stl_file(std::string fname,Real outpx,Real outpy,Real outpz)
    {
        std::string tmpline,tmp1,tmp2;
        int nlines=0;

        m_outpx=outpx;
        m_outpy=outpy;
        m_outpz=outpz;

        Vector<char> fileCharPtr;
        ParallelDescriptor::ReadAndBcastFile(fname, fileCharPtr);
        std::string fileCharPtrString(fileCharPtr.dataPtr());
        std::istringstream infile(fileCharPtrString, std::istringstream::in);

        if(amrex::Verbose())
            Print()<<"STL file name:"<<fname<<"\n";

        std::getline(infile,tmpline); //solid <solidname>
        while(!infile.eof())
        {
            std::getline(infile,tmpline);
            if(tmpline.find("endsolid")!=std::string::npos)
            {
                break;
            }
            nlines++;
        }

        if(nlines%m_nlines_per_facet!=0)
        {
            Abort("may be there are blank lines in the STL file\n");
        }

        m_num_tri=nlines/m_nlines_per_facet;

        if(amrex::Verbose())
            Print()<<"number of triangles:"<<m_num_tri<<"\n";

        //host vectors
        m_tri_pts_h.resize(m_num_tri*m_ndata_per_tri);
        m_tri_normals_h.resize(m_num_tri*m_ndata_per_normal);

        infile.seekg(0);
        std::getline(infile,tmpline); //solid <solidname>

        for(int i=0;i<m_num_tri;i++)
        {
            std::getline(is,tmp); // facet normal
            std::getline(is,tmp); // outer loop

            Real x, y, z;

            for (int iv = 0; iv < 3; ++iv) { // 3 vertices
                is >> tmp >> x >> y >> z;
                *p++ = x * scale + center[0];
                *p++ = y * scale + center[1];
                *p++ = z * scale + center[2];
            }
            std::getline(is,tmp); // read \n

            std::getline(is,tmp); //end loop
            std::getline(is,tmp); //end facet

            if (reverse_normal) {
                std::swap(m_tri_pts_h[i].v1, m_tri_pts_h[i].v2);
            }
        }
    }
}

void
STLtools::prepare ()
{
    ParallelDescriptor::Bcast(&m_num_tri, 1);
    if (!ParallelDescriptor::IOProcessor()) {
        m_tri_pts_h.resize(m_num_tri);
    }
    ParallelDescriptor::Bcast((char*)(m_tri_pts_h.dataPtr()), m_num_tri*sizeof(Triangle));

    //device vectors
    m_tri_pts_d.resize(m_num_tri);
    m_tri_normals_d.resize(m_num_tri);

    Gpu::copyAsync(Gpu::hostToDevice, m_tri_pts_h.begin(), m_tri_pts_h.end(),
                   m_tri_pts_d.begin());

    Triangle const* tri_pts = m_tri_pts_d.data();
    XDim3* tri_norm = m_tri_normals_d.data();


    // Compute normals in case the STL file does not have valid data for normals
    m_triangles.clear();
    ParallelFor(m_num_tri, [=] AMREX_GPU_DEVICE (int i) noexcept
    {
        Triangle const& tri = tri_pts[i];
        XDim3 vec1{tri.v2.x-tri.v1.x, tri.v2.y-tri.v1.y, tri.v2.z-tri.v1.z};
        XDim3 vec2{tri.v3.x-tri.v2.x, tri.v3.y-tri.v2.y, tri.v3.z-tri.v2.z};
        XDim3 norm{vec1.y*vec2.z-vec1.z*vec2.y,
                   vec1.z*vec2.x-vec1.x*vec2.z,
                   vec1.x*vec2.y-vec1.y*vec2.x};
        Real tmp = 1._rt / std::sqrt(norm.x*norm.x + norm.y*norm.y + norm.z*norm.z);
        tri_norm[i].x = norm.x * tmp;
        tri_norm[i].y = norm.y * tmp;
        tri_norm[i].z = norm.z * tmp;

        // EY: Make a list of TriangleC
        Point a(tri.v1.x, tri.v1.y, tri.v1.z);
        Point b(tri.v2.x, tri.v2.y, tri.v2.z);
        Point c(tri.v3.x, tri.v3.y, tri.v3.z);

        m_triangles.push_back(TriangleC(a,b,c));
    });
    m_aabb_tree = new Tree(m_triangles.begin(),m_triangles.end());


    ReduceOps<ReduceOpMin,ReduceOpMin,ReduceOpMin,ReduceOpMax,ReduceOpMax,ReduceOpMax> reduce_op;
    ReduceData<Real,Real,Real,Real,Real,Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;
    reduce_op.eval(m_num_tri, reduce_data,
                   [=] AMREX_GPU_DEVICE (int i) -> ReduceTuple
                   {
                       return {amrex::min(tri_pts[i].v1.x,
                                          tri_pts[i].v2.x,
                                          tri_pts[i].v3.x),
                               amrex::min(tri_pts[i].v1.y,
                                          tri_pts[i].v2.y,
                                          tri_pts[i].v3.y),
                               amrex::min(tri_pts[i].v1.z,
                                          tri_pts[i].v2.z,
                                          tri_pts[i].v3.z),
                               amrex::max(tri_pts[i].v1.x,
                                          tri_pts[i].v2.x,
                                          tri_pts[i].v3.x),
                               amrex::max(tri_pts[i].v1.y,
                                          tri_pts[i].v2.y,
                                          tri_pts[i].v3.y),
                               amrex::max(tri_pts[i].v1.z,
                                          tri_pts[i].v2.z,
                                          tri_pts[i].v3.z)};
                   });
    auto const& hv = reduce_data.value(reduce_op);
    m_ptmin.x = amrex::get<0>(hv);
    m_ptmin.y = amrex::get<1>(hv);
    m_ptmin.z = amrex::get<2>(hv);
    m_ptmax.x = amrex::get<3>(hv);
    m_ptmax.y = amrex::get<4>(hv);
    m_ptmax.z = amrex::get<5>(hv);

    if (amrex::Verbose() > 0) {
        amrex::Print() << "    Min: " << m_ptmin << " Max: " << m_ptmax << '\n';
    }

    // Choose a reference point by extending the normal vector of the first
    // triangle until it's slightly outside the bounding box.
    XDim3 cent0; // centroid of the first triangle
    int is_ref_positive;
    {
        Triangle const& tri = m_tri_pts_h[0];
        cent0 = XDim3{(tri.v1.x + tri.v2.x + tri.v3.x) / 3._rt,
                      (tri.v1.y + tri.v2.y + tri.v3.y) / 3._rt,
                      (tri.v1.z + tri.v2.z + tri.v3.z) / 3._rt};
        // We are computing the normal ourselves in case the stl file does
        // not have valid data on normal.
        XDim3 vec1{tri.v2.x-tri.v1.x, tri.v2.y-tri.v1.y, tri.v2.z-tri.v1.z};
        XDim3 vec2{tri.v3.x-tri.v2.x, tri.v3.y-tri.v2.y, tri.v3.z-tri.v2.z};
        XDim3 norm{vec1.y*vec2.z-vec1.z*vec2.y,
                   vec1.z*vec2.x-vec1.x*vec2.z,
                   vec1.x*vec2.y-vec1.y*vec2.x};
        Real tmp = 1._rt / std::sqrt(norm.x*norm.x + norm.y*norm.y + norm.z*norm.z);
        norm.x *= tmp;
        norm.y *= tmp;
        norm.z *= tmp;
        // Now we need to find out where the normal vector will intersect
        // with the bounding box defined by m_ptmin and m_ptmax.
        Real Lx, Ly, Lz;
        constexpr Real eps = std::numeric_limits<Real>::epsilon();
        if (norm.x > eps) {
            Lx = (m_ptmax.x-cent0.x) / norm.x;
        } else if (norm.x < -eps) {
            Lx = (m_ptmin.x-cent0.x) / norm.x;
        } else {
            Lx = std::numeric_limits<Real>::max();
        }
        if (norm.y > eps) {
            Ly = (m_ptmax.y-cent0.y) / norm.y;
        } else if (norm.y < -eps) {
            Ly = (m_ptmin.y-cent0.y) / norm.y;
        } else {
            Ly = std::numeric_limits<Real>::max();
        }
        if (norm.z > eps) {
            Lz = (m_ptmax.z-cent0.z) / norm.z;
        } else if (norm.z < -eps) {
            Lz = (m_ptmin.z-cent0.z) / norm.z;
        } else {
            Lz = std::numeric_limits<Real>::max();
        }
        Real Lp = std::min({Lx,Ly,Lz});
        if (norm.x > eps) {
            Lx = (m_ptmin.x-cent0.x) / norm.x;
        } else if (norm.x < -eps) {
            Lx = (m_ptmax.x-cent0.x) / norm.x;
        } else {
            Lx = std::numeric_limits<Real>::lowest();
        }
        if (norm.y > eps) {
            Ly = (m_ptmin.y-cent0.y) / norm.y;
        } else if (norm.y < -eps) {
            Ly = (m_ptmax.y-cent0.y) / norm.y;
        } else {
            Ly = std::numeric_limits<Real>::lowest();
        }
        if (norm.z > eps) {
            Lz = (m_ptmin.z-cent0.z) / norm.z;
        } else if (norm.z < -eps) {
            Lz = (m_ptmax.z-cent0.z) / norm.z;
        } else {
            Lz = std::numeric_limits<Real>::lowest();
        }
        if (std::abs(norm.x) < 1.e-5) {
            norm.x = std::copysign(Real(1.e-5), norm.x);
        }
        if (std::abs(norm.y) < 1.e-5) {
            norm.y = std::copysign(Real(1.e-5), norm.y);
        }
        if (std::abs(norm.z) < 1.e-5) {
            norm.z = std::copysign(Real(1.e-5), norm.z);
        }
        Real Lm = std::max({Lx,Ly,Lz});
        Real Leps = std::max(Lp,-Lm) * Real(0.009);
        if (Lp < -Lm) {
            m_ptref.x = cent0.x + (Lp+Leps) * norm.x;
            m_ptref.y = cent0.y + (Lp+Leps) * norm.y;
            m_ptref.z = cent0.z + (Lp+Leps) * norm.z;
            is_ref_positive = true;
        } else {
            m_ptref.x = cent0.x + (Lm-Leps) * norm.x;
            m_ptref.y = cent0.y + (Lm-Leps) * norm.y;
            m_ptref.z = cent0.z + (Lm-Leps) * norm.z;
            is_ref_positive = false;
        }
    }

    // We now need to figure out if the boundary and the reference is
    // outside or inside the object.
    XDim3 ptref = m_ptref;
    int num_isects = Reduce::Sum<int>(m_num_tri, [=] AMREX_GPU_DEVICE (int i) -> int
        {
            if (i == 0) {
                return 1-is_ref_positive;
            } else {
                Real p1[] = {ptref.x, ptref.y, ptref.z};
                Real p2[] = {cent0.x, cent0.y, cent0.z};
                return static_cast<int>(line_tri_intersects(p1, p2, tri_pts[i]));
            }
        });

    m_boundry_is_outside = num_isects % 2 == 0;
}

void
STLtools::fill (MultiFab& mf, IntVect const& nghost, Geometry const& geom,
                Real outside_value, Real inside_value) const
{
    int num_triangles = m_num_tri;

    const auto plo = geom.ProbLoArray();
    const auto dx  = geom.CellSizeArray();

    const Triangle* tri_pts = m_tri_pts_d.data();
    XDim3 ptmin = m_ptmin;
    XDim3 ptmax = m_ptmax;
    XDim3 ptref = m_ptref;
    Real reference_value = m_boundry_is_outside ? outside_value :  inside_value;
    Real other_value     = m_boundry_is_outside ?  inside_value : outside_value;
    
    // EY: checking point
    // amrex::Print() << "reference_value = " << reference_value << "\n";
    // amrex::Print() << "other_value = " << other_value << "\n";


    auto const& ma = mf.arrays();

    ParallelFor(mf, nghost, [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
    {
        Real coords[3];
        coords[0]=plo[0]+static_cast<Real>(i)*dx[0];
        coords[1]=plo[1]+static_cast<Real>(j)*dx[1];
#if (AMREX_SPACEDIM == 2)
        coords[2]=Real(0.);
#else
        coords[2]=plo[2]+static_cast<Real>(k)*dx[2];
#endif
        int num_intersects=0;

        if (coords[0] >= ptmin.x && coords[0] <= ptmax.x &&
            coords[1] >= ptmin.y && coords[1] <= ptmax.y &&
            coords[2] >= ptmin.z && coords[2] <= ptmax.z)
        {
            Real pr[]={ptref.x, ptref.y, ptref.z};

            // EY: Use CGAL aabb tree 
            num_intersects = getNumIntersect(coords[0], coords[1], coords[2], ptref.x, ptref.y, ptref.z);
            
            // // //Original line search for ALL triangle
            // for (int tr=0; tr < num_triangles; ++tr) {
            //     if (line_tri_intersects(pr, coords, tri_pts[tr])) {
            //         ++num_intersects;
            //     }
            // }

        }
        // amrex::Print() << "number of intersections = " << num_intersects << "\n";
        ma[box_no](i,j,k) = (num_intersects % 2 == 0) ? reference_value : other_value;
    });
    // auto t1 = std::chrono::high_resolution_clock::now();
    // auto dt = 1.e-9*std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();
    // amrex::Print() << "Time for line-tri-intersects = " << dt << "(s)" << "\n";

    Gpu::streamSynchronize();
}

int
STLtools::getBoxType (Box const& box, Geometry const& geom, RunOn) const
{
    const auto plo = geom.ProbLoArray();
    const auto dx  = geom.CellSizeArray();

    XDim3 blo{plo[0] + static_cast<Real>(box.smallEnd(0))*dx[0],
              plo[1] + static_cast<Real>(box.smallEnd(1))*dx[1],
#if (AMREX_SPACEDIM == 2)
              0._rt
#else
              plo[2] + static_cast<Real>(box.smallEnd(2))*dx[2]
#endif
    };

    XDim3 bhi{plo[0] + static_cast<Real>(box.bigEnd(0))*dx[0],
              plo[1] + static_cast<Real>(box.bigEnd(1))*dx[1],
#if (AMREX_SPACEDIM == 2)
              0._rt
#else
              plo[2] + static_cast<Real>(box.bigEnd(2))*dx[2]
#endif
    };

    if (blo.x > m_ptmax.x || blo.y > m_ptmax.y || blo.z > m_ptmax.z ||
        bhi.x < m_ptmin.x || bhi.y < m_ptmin.y || bhi.z < m_ptmin.z)
    {
        return m_boundry_is_outside ? allregular : allcovered;
    }
    else
    {
        int num_triangles = m_num_tri;
        const Triangle* tri_pts = m_tri_pts_d.data();
        XDim3 ptmin = m_ptmin;
        XDim3 ptmax = m_ptmax;
        XDim3 ptref = m_ptref;
        int ref_value = m_boundry_is_outside ? 1 : 0;

        // EY: Timing for line-search
        // auto t0 = std::chrono::high_resolution_clock::now();  

        ReduceOps<ReduceOpSum> reduce_op;
        ReduceData<int> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
        reduce_op.eval(box, reduce_data,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
        {
            Real coords[3];
            coords[0]=plo[0]+static_cast<Real>(i)*dx[0];
            coords[1]=plo[1]+static_cast<Real>(j)*dx[1];
#if (AMREX_SPACEDIM == 2)
            amrex::ignore_unused(k);
            coords[2]=Real(0.);
#else
            coords[2]=plo[2]+static_cast<Real>(k)*dx[2];
#endif
            int num_intersects=0; 
            if (coords[0] >= ptmin.x && coords[0] <= ptmax.x &&
                coords[1] >= ptmin.y && coords[1] <= ptmax.y &&
                coords[2] >= ptmin.z && coords[2] <= ptmax.z)
            {
                Real pr[]={ptref.x, ptref.y, ptref.z};

                // EY: Use CGAL aabb tree 
                num_intersects = getNumIntersect(coords[0], coords[1], coords[2], ptref.x, ptref.y, ptref.z);

                // EY: replace this with CGAL
                // for (int tr=0; tr < num_triangles; ++tr) {
                //     if (line_tri_intersects(pr, coords, tri_pts[tr])) {
                //         ++num_intersects;
                //     }
                // } 
            }
            return (num_intersects % 2 == 0) ? ref_value : 1-ref_value;
        });
        // auto t1 = std::chrono::high_resolution_clock::now();
        // auto dt = 1.e-9*std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();
        // amrex::Print() << "Time for line-tri-intersects = " << dt << "(s)" << "\n";

        ReduceTuple hv = reduce_data.value(reduce_op);
        Long nfluid = static_cast<Long>(amrex::get<0>(hv));
        Long npts = box.numPts();
         
        if (nfluid == 0) {
            return allcovered;
        } else if (nfluid == npts) {
            return allregular;
        } else {
            return mixedcells;
        }
    }
}

void
STLtools::fillFab (BaseFab<Real>& levelset, const Geometry& geom, RunOn, Box const&) const
{   
    int num_triangles = m_num_tri;

    const auto plo = geom.ProbLoArray();
    const auto dx  = geom.CellSizeArray();

    const Triangle* tri_pts = m_tri_pts_d.data();
    XDim3 ptmin = m_ptmin;
    XDim3 ptmax = m_ptmax;
    XDim3 ptref = m_ptref;
    Real reference_value = m_boundry_is_outside ? -1.0_rt :  1.0_rt;
    Real other_value     = m_boundry_is_outside ?  1.0_rt : -1.0_rt;

    auto const& a = levelset.array();
    // EY: check getSingedDist
    auto const& a2 = levelset.array();
    const Box& bx = levelset.box();
    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        Real coords[3];
        coords[0]=plo[0]+static_cast<Real>(i)*dx[0];
        coords[1]=plo[1]+static_cast<Real>(j)*dx[1];
#if (AMREX_SPACEDIM == 2)
        coords[2]=Real(0.);
#else
        coords[2]=plo[2]+static_cast<Real>(k)*dx[2];
#endif
        int num_intersects = 0;
        Real signed_dist = 0;
        if (coords[0] >= ptmin.x && coords[0] <= ptmax.x &&
            coords[1] >= ptmin.y && coords[1] <= ptmax.y &&
            coords[2] >= ptmin.z && coords[2] <= ptmax.z)
        {
            Real pr[]={ptref.x, ptref.y, ptref.z};
            // EY: Use CGAL aabb tree 
            num_intersects = getNumIntersect(coords[0], coords[1], coords[2], ptref.x, ptref.y, ptref.z);

            signed_dist = getSignedDistance(coords[0], coords[1], coords[2]);

            // for (int tr=0; tr < num_triangles; ++tr) {
            //     if (line_tri_intersects(pr, coords, tri_pts[tr])) {
            //         ++num_intersects;
            //     }
            // }
        }
        a(i,j,k) = (num_intersects % 2 == 0) ? reference_value : other_value;
        a2(i,j,k) = signed_dist;
    });
}

void
STLtools::getIntercept (Array<Array4<Real>,AMREX_SPACEDIM> const& inter_arr,
                        Array<Array4<EB2::Type_t const>,AMREX_SPACEDIM> const& type_arr,
                        Array4<Real const> const& lst ,Geometry const& geom,
                        RunOn, Box const&) const
{
    int num_triangles = m_num_tri;

    const auto plo = geom.ProbLoArray();
    const auto dx  = geom.CellSizeArray();

    const Triangle* tri_pts = m_tri_pts_d.data();
    const XDim3* tri_norm = m_tri_normals_d.data();

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        Array4<Real> const& inter = inter_arr[idim];
        Array4<EB2::Type_t const> const& type = type_arr[idim];
        const Box bx{inter};
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real r = std::numeric_limits<Real>::quiet_NaN();
            if (type(i,j,k) == EB2::Type::irregular) {
                XDim3 p1{plo[0]+static_cast<Real>(i)*dx[0],
                         plo[1]+static_cast<Real>(j)*dx[1],
#if (AMREX_SPACEDIM == 2)
                         Real(0.)
#else
                         plo[2]+static_cast<Real>(k)*dx[2]
#endif
                };
                if (idim == 0) {
                    Real x2 = plo[0]+static_cast<Real>(i+1)*dx[0];
                    int it;
                    for (it=0; it < num_triangles; ++it) {
                        auto const& tri = tri_pts[it];
                        auto tmp = edge_tri_intersects(p1.x, x2, p1.y, p1.z,
                                                       tri.v1, tri.v2, tri.v3,
                                                       tri_norm[it],
                                                       lst(i+1,j,k)-lst(i,j,k));
                        if (tmp.first) {
                            r = tmp.second;
                            break;
                        }
                    }
                    if (it == num_triangles) {
                        r = (lst(i,j,k) > 0._rt) ? p1.x : x2;
                    }
                } else if (idim == 1) {
                    Real y2 = plo[1]+static_cast<Real>(j+1)*dx[1];
                    int it;
                    for (it=0; it < num_triangles; ++it) {
                        auto const& tri = tri_pts[it];
                        auto const& norm = tri_norm[it];
                        auto tmp = edge_tri_intersects(p1.y, y2, p1.z, p1.x,
                                                       {tri.v1.y, tri.v1.z, tri.v1.x},
                                                       {tri.v2.y, tri.v2.z, tri.v2.x},
                                                       {tri.v3.y, tri.v3.z, tri.v3.x},
                                                       {  norm.y,   norm.z,   norm.x},
                                                       lst(i,j+1,k)-lst(i,j,k));
                        if (tmp.first) {
                            r = tmp.second;
                            break;
                        }
                    }
                    if (it == num_triangles) {
                        r = (lst(i,j,k) > 0._rt) ? p1.y : y2;
                    }
                } else {
                    Real z2 = plo[2]+static_cast<Real>(k+1)*dx[2];
                    int it;
                    for (it=0; it < num_triangles; ++it) {
                        auto const& tri = tri_pts[it];
                        auto const& norm = tri_norm[it];
                        auto tmp = edge_tri_intersects(p1.z, z2, p1.x, p1.y,
                                                       {tri.v1.z, tri.v1.x, tri.v1.y},
                                                       {tri.v2.z, tri.v2.x, tri.v2.y},
                                                       {tri.v3.z, tri.v3.x, tri.v3.y},
                                                       {  norm.z,   norm.x,   norm.y},
                                                       lst(i,j,k+1)-lst(i,j,k));
                        if (tmp.first) {
                            r = tmp.second;
                            break;
                        }
                    }
                    if (it == num_triangles) {
                        r = (lst(i,j,k) > 0._rt) ? p1.z : z2;
                    }
                }
            }
            inter(i,j,k) = r;
        });
    }
}

void
STLtools::updateIntercept (Array<Array4<Real>,AMREX_SPACEDIM> const& inter_arr,
                           Array<Array4<EB2::Type_t const>,AMREX_SPACEDIM> const& type_arr,
                           Array4<Real const> const& lst, Geometry const& geom)
{
    auto const& dx = geom.CellSizeArray();
    auto const& problo = geom.ProbLoArray();
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        Array4<Real> const& inter = inter_arr[idim];
        Array4<EB2::Type_t const> const& type = type_arr[idim];
        const Box bx{inter};
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if (type(i,j,k) == EB2::Type::irregular) {
                bool is_nan = amrex::isnan(inter(i,j,k));
                if (idim == 0) {
                    if (lst(i,j,k) == Real(0.0) ||
                        (lst(i,j,k) > Real(0.0) && is_nan))
                    {
                        // interp might still be quiet_nan because lst that
                        // was set to zero has been changed by FillBoundary
                        // at periodic boundaries.
                        inter(i,j,k) = problo[0] + static_cast<Real>(i)*dx[0];
                    }
                    else if (lst(i+1,j,k) == Real(0.0) ||
                             (lst(i+1,j,k) > Real(0.0) && is_nan))
                    {
                        inter(i,j,k) = problo[0] + static_cast<Real>(i+1)*dx[0];
                    }
                } else if (idim == 1) {
                    if (lst(i,j,k) == Real(0.0) ||
                        (lst(i,j,k) > Real(0.0) && is_nan))
                    {
                        inter(i,j,k) = problo[1] + static_cast<Real>(j)*dx[1];
                    }
                    else if (lst(i,j+1,k) == Real(0.0) ||
                             (lst(i,j+1,k) > Real(0.0) && is_nan))
                    {
                        inter(i,j,k) = problo[1] + static_cast<Real>(j+1)*dx[1];
                    }
                } else {
                    if (lst(i,j,k) == Real(0.0) ||
                        (lst(i,j,k) > Real(0.0) && is_nan))
                    {
                        inter(i,j,k) = problo[2] + static_cast<Real>(k)*dx[2];
                        }
                    else if (lst(i,j,k+1) == Real(0.0) ||
                             (lst(i,j,k+1) > Real(0.0) && is_nan))
                    {
                        inter(i,j,k) = problo[2] + static_cast<Real>(k+1)*dx[2];
                    }
                }
            }
        });
    }
}

}
