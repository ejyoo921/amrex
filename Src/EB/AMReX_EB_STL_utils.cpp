#include<AMReX_EB_STL_utils.H>
#include<AMReX_EB_triGeomOps_K.H>

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
    });

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

        Point crd(x,y,z);

        //FIXME: get a point outside from user
        Point point_outside(m_outpx,m_outpy,m_outpz);
        Segment out_to_coord(point_outside,crd);
        num_intersects=(*m_aabb_tree).number_of_intersected_primitives(out_to_coord);

        sign=(num_intersects%2==0)?1.0:-1.0;
    
        Point closest_point = (*m_aabb_tree).closest_point(crd);
        FT sqd = (*m_aabb_tree).squared_distance(crd);
        // dist = sqrt(sqd)*sign;
        dist = sqd*sign;

        return(dist);
    }
    //================================================================================
    /*
    void STLtools::stl_to_markerfab(MultiFab& markerfab,Geometry geom,
            Real outpx,Real outpy,Real outpz)
    {
        //local variables for lambda capture
        int data_stride   = m_ndata_per_tri;
        int num_triangles = m_num_tri;
        Real outvalue     = m_outside;
        Real invalue      = m_inside;

        const auto plo   = geom.ProbLoArray();
        const auto dx    = geom.CellSizeArray();
        GpuArray<Real,3> outp={outpx,outpy,outpz};

        const Real *tri_pts=m_tri_pts_d.data();

        for (MFIter mfi(markerfab); mfi.isValid(); ++mfi) // Loop over grids
        {
            const Box& bx = mfi.validbox();
            auto mfab_arr=markerfab[mfi].array();

            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                Real coords[3],po[3];
                Real t1[3],t2[3],t3[3];

                coords[0]=plo[0]+i*dx[0];
                coords[1]=plo[1]+j*dx[1];
                coords[2]=plo[2]+k*dx[2];

                po[0]=outp[0];
                po[1]=outp[1];
                po[2]=outp[2];

                int num_intersects=0;
                int intersect;

                for(int tr=0;tr<num_triangles;tr++)
                {
                    t1[0]=tri_pts[tr*data_stride+0];
                    t1[1]=tri_pts[tr*data_stride+1];
                    t1[2]=tri_pts[tr*data_stride+2];

                    t2[0]=tri_pts[tr*data_stride+3];
                    t2[1]=tri_pts[tr*data_stride+4];
                    t2[2]=tri_pts[tr*data_stride+5];

                    t3[0]=tri_pts[tr*data_stride+6];
                    t3[1]=tri_pts[tr*data_stride+7];
                    t3[2]=tri_pts[tr*data_stride+8];

                    intersect = tri_geom_ops::lineseg_tri_intersect(po,coords,t1,t2,t3);
                    num_intersects += (1-intersect);
                }
                if(num_intersects%2 == 0)
                {
                    mfab_arr(i,j,k)=outvalue;
                }
                else
                {
                    mfab_arr(i,j,k)=invalue;
                }

            });
        }
    }
    //================================================================================
    */
}
