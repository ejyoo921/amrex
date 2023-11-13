#ifndef AMREX_EB_STL_UTILS_H_
#define AMREX_EB_STL_UTILS_H_

#include <AMReX_Config.H>
#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Box.H>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>

typedef CGAL::Simple_cartesian<double> K;
typedef K::FT FT;
typedef K::Ray_3 Ray;
typedef K::Line_3 Line;
typedef K::Segment_3 Segment;
typedef K::Point_3 Point;
typedef K::Triangle_3 Triangle;
typedef std::list<Triangle>::iterator Iterator;
typedef CGAL::AABB_triangle_primitive<K, Iterator> Primitive;
typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;
typedef Tree::Point_and_primitive_id Point_and_primitive_id;

namespace amrex
{
    class STLtools
    {
        private:

            //host vectors
            Gpu::PinnedVector<Real> m_tri_pts_h;
            Gpu::PinnedVector<Real> m_tri_normals_h;

            //device vectors
            Gpu::DeviceVector<amrex::Real> m_tri_pts_d;
            Gpu::DeviceVector<amrex::Real> m_tri_normals_d;

            int  m_num_tri=0;
            int  m_ndata_per_tri=9;    //three points x 3 coordinates
            int  m_ndata_per_normal=3; //three components
            int  m_nlines_per_facet=7; //specific to ASCII STLs
            Real m_inside  = -1.0;
            Real m_outside =  1.0;
        
            std::list<Triangle> m_triangles;
            Tree *m_aabb_tree;

            Real m_outpx = 1e5;
            Real m_outpy = 1e5;
            Real m_outpz = 1e5;

        public:

            void read_ascii_stl_file(std::string fname,Real outpx,Real outpy,Real outpz);
            // void stl_to_markerfab(MultiFab& markerfab,
            //         Geometry geom,Real outpx,Real outpy,Real outpz);
            Real getSignedDistance(Real x,Real y,Real z);

    };
}
#endif
