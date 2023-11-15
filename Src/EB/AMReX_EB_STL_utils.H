#ifndef AMREX_EB_STL_UTILS_H_
#define AMREX_EB_STL_UTILS_H_

#include <AMReX_Config.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Dim3.H>
#include <AMReX_EB2_Graph.H>

//EY from Hari - for CGAL
#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
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
typedef K::Triangle_3 TriangleC;

// using namespace std;
typedef std::list<TriangleC>::iterator Iterator;
typedef CGAL::AABB_triangle_primitive<K, Iterator> Primitive;
typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;
typedef Tree::Point_and_primitive_id Point_and_primitive_id;


namespace amrex
{

class STLtools
{
public:
    struct Triangle {
        XDim3 v1, v2, v3;
    };

    static constexpr int allregular = -1;
    static constexpr int mixedcells = 0;
    static constexpr int allcovered = 1;

private:

    Gpu::PinnedVector<Triangle> m_tri_pts_h;
    Gpu::DeviceVector<Triangle> m_tri_pts_d;
    Gpu::DeviceVector<XDim3> m_tri_normals_d;

    int m_num_tri=0;

    XDim3 m_ptmin;  // All triangles are inside the bounding box defined by
    XDim3 m_ptmax;  //     m_ptmin and m_ptmax.
    XDim3 m_ptref;  // The reference point is slightly outside the bounding box.
    bool m_boundry_is_outside; // Is the bounding box boundary outside or inside the object?

    void read_ascii_stl_file (std::string const& fname, Real scale,
                              Array<Real,3> const& center, int reverse_normal);
    void read_binary_stl_file (std::string const& fname, Real scale,
                               Array<Real,3> const& center, int reverse_normal);


    // EY from Hari for CGAL
    // //host vectors
    Gpu::PinnedVector<Real> m_tri_normals_h;

    // //device vectors
    // Gpu::DeviceVector<amrex::Real> m_tri_normals_d;

    // int  m_ndata_per_tri=9;    //three points x 3 coordinates
    // int  m_ndata_per_normal=3; //three components
    // int  m_nlines_per_facet=7; //specific to ASCII STLs
    // Real m_inside  = -1.0;
    // Real m_outside =  1.0;

    int  m_ndata_per_tri = 9;    //three points x 3 coordinates
    std::list<TriangleC> m_triangles;
    Tree *m_aabb_tree;

    Real m_outpx = 1e5;
    Real m_outpy = 1e5;
    Real m_outpz = 1e5;

public:

    void prepare ();  // public for cuda

    // EY from Hari
    Real getSignedDistance(Real x, Real y, Real z) const;
    Real getNumIntersect(Real x, Real y, Real z, Real outpx, Real outpy, Real outpz) const;



    void read_stl_file (std::string const& fname, Real scale, Array<Real,3> const& center,
                        int reverse_normal);

    void fill (MultiFab& mf, IntVect const& nghost, Geometry const& geom,
               Real outside_value = -1._rt, Real inside_value = 1._rt) const;

    [[nodiscard]] int getBoxType (Box const& box, Geometry const& geom, RunOn) const;

    static constexpr bool isGPUable () noexcept { return true; }

    void fillFab (BaseFab<Real>& levelset, const Geometry& geom, RunOn,
                  Box const& bounding_box) const;

    void getIntercept (Array<Array4<Real>,AMREX_SPACEDIM> const& inter_arr,
                       Array<Array4<EB2::Type_t const>,AMREX_SPACEDIM> const& type_arr,
                       Array4<Real const> const& lst, Geometry const& geom, RunOn,
                       Box const& bounding_box) const;

    static void updateIntercept (Array<Array4<Real>,AMREX_SPACEDIM> const& inter_arr,
                          Array<Array4<EB2::Type_t const>,AMREX_SPACEDIM> const& type_arr,
                          Array4<Real const> const& lst, Geometry const& geom) ;

};

}
#endif
