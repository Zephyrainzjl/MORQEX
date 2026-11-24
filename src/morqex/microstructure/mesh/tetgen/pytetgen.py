"""Python module to interface with wrapped TetGen C++ code
"""
import ctypes

import numpy as np
from . import _tetgen




invalid_input = TypeError(
    "Invalid input.  Must be either a pyvista.PolyData\n" + "object or vertex and face arrays"
)


class TetGen:
    """Input, clean, and tetrahedralize surface meshes using TetGen.
    """

    _updated = None

    def __init__(self, *args):
        """initializes MeshFix using a mesh"""
        self.v = None
        self.f = None
        self.node = None
        self.elem = None
        self._grid = None

        if not args:
            raise invalid_input
        elif isinstance(args[0], (np.ndarray, list)):
            self._load_arrays(args[0], args[1])
        else:
            raise invalid_input

    def _load_arrays(self, v, f):
        """Loads triangular mesh from vertex and face arrays

        Face arrays/lists are v and f.  Both vertex and face arrays
        should be 2D arrays with each vertex containing XYZ data and
        each face containing three points.
        """
        # Check inputs
        if not isinstance(v, np.ndarray):
            try:
                v = np.asarray(v, np.floating)
                if v.ndim != 2 and v.shape[1] != 3:
                    raise Exception("Invalid vertex format.  Shape should be (npoints, 3)")
            except BaseException:
                raise Exception("Unable to convert vertex input to valid numpy array")

        if not isinstance(f, np.ndarray):
            try:
                f = np.asarray(f, ctypes.c_int)
                if f.ndim != 2 and f.shape[1] != 3:
                    raise Exception("Invalid face format.  Shape should be (nfaces, 3)")
            except BaseException:
                raise Exception("Unable to convert face input to valid numpy array")

        # Store to self
        self.v = v
        self.f = f


    def tetrahedralize(
        self,
        plc=True,
        psc=0.0,
        refine=0.0,
        quality=True,
        nobisect=False,
        cdt=0.0,
        cdtrefine=7.0,
        coarsen=0.0,
        weighted=0.0,
        brio_hilbert=1.0,
        flipinsert=0.0,
        metric=0.0,
        varvolume=0.0,
        fixedvolume=0.0,
        regionattrib=0.0,
        insertaddpoints=0.0,
        diagnose=0.0,
        convex=0.0,
        nomergefacet=0.0,
        nomergevertex=0.0,
        noexact=0.0,
        nostaticfilter=0.0,
        zeroindex=0.0,
        facesout=0.0,
        edgesout=0.0,
        neighout=0.0,
        voroout=0.0,
        meditview=0.0,
        vtkview=0.0,
        vtksurfview=0.0,
        nobound=0.0,
        nonodewritten=0.0,
        noelewritten=0.0,
        nofacewritten=0.0,
        noiterationnum=0.0,
        nojettison=0.0,
        docheck=0.0,
        quiet=0.0,
        nowarning=0.0,
        verbose=0.0,
        vertexperblock=4092.0,
        tetrahedraperblock=8188.0,
        shellfaceperblock=2044.0,
        supsteiner_level=2.0,
        addsteiner_algo=1.0,
        coarsen_param=0.0,
        weighted_param=0.0,
        fliplinklevel=-1.0,
        flipstarsize=-1.0,
        fliplinklevelinc=1.0,
        opt_max_flip_level=3.0,
        opt_scheme=7.0,
        opt_iterations=3.0,
        smooth_cirterion=1.0,
        smooth_maxiter=7.0,
        delmaxfliplevel=1.0,
        order=1.0,
        reversetetori=0.0,
        steinerleft=100000.0,
        unflip_queue_limit=1000.0,
        no_sort=0.0,
        hilbert_order=52.0,
        hilbert_limit=8.0,
        brio_threshold=64.0,
        brio_ratio=0.125,
        epsilon=1.0e-8,
        facet_separate_ang_tol=179.9,
        collinear_ang_tol=179.9,
        facet_small_ang_tol=15.0,
        maxvolume=-1.0,
        maxvolume_length=-1.0,
        minratio=2.0,
        opt_max_asp_ratio=1000.0,
        opt_max_edge_ratio=100.0,
        mindihedral=0.0,
        optmaxdihedral=177.0,
        metric_scale=1.0,
        smooth_alpha=0.3,
        coarsen_percent=1.0,
        elem_growth_ratio=0.0,
        refine_progress_ratio=0.333,
        switches=None,
    ):
        """Generates tetrahedrals interior to the surface mesh
        described by the vertex and face arrays already loaded.
        Returns nodes and elements belonging to the all tetrahedral
        mesh.

        The tetrahedral generator uses the C++ library TetGen and can
        be configured by either using a string of ``switches`` or by
        changing the underlying behavior using optional inputs.

        Should the user desire more control over the mesh
        tetrahedralization or wish to control the tetrahedralization
        in a more pythonic manner, use the optional inputs rather than
        inputting switches.

        Parameters
        ----------
        quality : bool, optional
            Enables/disables mesh improvement.  Enabled by default.
            Disable this to speed up mesh generation while sacrificing
            quality.  Default True.

        minratio : double, optional.
            Maximum allowable radius-edge ratio.  Must be greater than
            1.0 the closer to 1.0, the higher the quality of the mesh.
            Be sure to raise ``steinerleft`` to allow for the addition of
            points to improve the quality of the mesh.  Avoid overly
            restrictive requirements, otherwise, meshing will appear
            to hang.  Default 2.0

            Testing has showed that 1.1 is a reasonable input for a
            high quality mesh.

        mindihedral : double, optional
            Minimum allowable dihedral angle.  The larger this number,
            the higher the quality of the resulting mesh.  Be sure to
            raise ``steinerleft`` to allow for the addition of points to
            improve the quality of the mesh.  Avoid overly restrictive
            requirements, otherwise, meshing will appear to hang.
            Default 0.0

            Testing has shown that 10 is a reasonable input

        verbose : int, optional
            Controls the underlying TetGen library to output text to
            console.  Users using iPython will not see this output.
            Setting to 1 enables some information about the mesh
            generation while setting verbose to 2 enables more debug
            output.  Default 0, or no output.

        nobisect : bool, optional
            Controls if Steiner points are added to the input surface
            mesh.  When enabled, the surface mesh will be modified.
            Default False.

            Testing has shown that if your input surface mesh is
            already well shaped, disabling this setting will improve
            meshing speed and mesh quality.

        steinerleft : int, optional
            Steiner points are points added to the original surface
            mesh to create a valid tetrahedral mesh.  Settings this to
            -1 will allow tetgen to create an unlimited number of
            steiner points, but the program will likely hang if this
            is used in combination with narrow quality requirements.
            Default 100000.

            The first type of Steiner points are used in creating an
            initial tetrahedralization of PLC. These Steiner points
            are mandatory in order to create a valid
            tetrahedralization

            The second type of Steiner points are used in creating
            quality tetra- hedral meshes of PLCs. These Steiner points
            are optional, while they may be necessary in order to
            improve the mesh quality or to conform the size of mesh
            elements.

        order : int optional
            Controls whether TetGen creates linear tetrahedrals or
            quadradic tetrahedrals.  Set order to 2 to output
            quadradic tetrahedrals.  Default 2.

        Examples
        --------
        The following switches "pq1.1/10Y" would be:

        >>> node, elem = tgen.tetrahedralize(nobisect=True, quality=True,
                                             minratio=1.1, mindihedral=10)

        Using the switches option:

        >>> node, elem = tgen.tetrahedralize(switches="pq1.1/10Y")

        Notes
        -----
        There are many other options and the TetGen documentation
        contains descriptions only for the switches of the original
        C++ program.  This is the relationship between tetgen switches
        and python optional inputs:

        Switches of TetGen.

        +---------------------------+---------------+---------+
        | Option                    | Switch        | Default |
        +---------------------------+---------------+---------+
        | plc                       | ``'-p'``      | 0.      |
        +---------------------------+---------------+---------+
        | psc                       | ``'-s'``      | 0.      |
        +---------------------------+---------------+---------+
        | refine                    | ``'-r'``      | 0.      |
        +---------------------------+---------------+---------+
        | quality                   | ``'-q'``      | 0.      |
        +---------------------------+---------------+---------+
        | nobisect                  | ``'-Y'``      | 0.      |
        +---------------------------+---------------+---------+
        | cdt                       | ``'-D'``      | 0.      |
        +---------------------------+---------------+---------+
        | cdtrefine                 | ``'-D#'``     | 7.      |
        +---------------------------+---------------+---------+
        | coarsen                   | ``'-R'``      | 0.      |
        +---------------------------+---------------+---------+
        | weighted                  | ``'-w'``      | 0.      |
        +---------------------------+---------------+---------+
        | brio_hilbert              | ``'-b'``      | 1.      |
        +---------------------------+---------------+---------+
        | flipinsert                | ``'-L'``      | 0.      |
        +---------------------------+---------------+---------+
        | metric                    | ``'-m'``      | 0.      |
        +---------------------------+---------------+---------+
        | varvolume                 | ``'-a'``      | 0.      |
        +---------------------------+---------------+---------+
        | fixedvolume               | ``'-a'``      | 0.      |
        +---------------------------+---------------+---------+
        | regionattrib              | ``'-A'``      | 0.      |
        +---------------------------+---------------+---------+
        | insertaddpoints           | ``'-i'``      | 0.      |
        +---------------------------+---------------+---------+
        | diagnose                  | ``'-d'``      | 0.      |
        +---------------------------+---------------+---------+
        | convex                    | ``'-c'``      | 0.      |
        +---------------------------+---------------+---------+
        | nomergefacet              | ``'-M'``      | 0.      |
        +---------------------------+---------------+---------+
        | nomergevertex             | ``'-M'``      | 0.      |
        +---------------------------+---------------+---------+
        | noexact                   | ``'-X'``      | 0.      |
        +---------------------------+---------------+---------+
        | nostaticfilter            | ``'-X'``      | 0.      |
        +---------------------------+---------------+---------+
        | zeroindex                 | ``'-z'``      | 0.      |
        +---------------------------+---------------+---------+
        | facesout                  | ``'-f'``      | 0.      |
        +---------------------------+---------------+---------+
        | edgesout                  | ``'-e'``      | 0.      |
        +---------------------------+---------------+---------+
        | neighout                  | ``'-n'``      | 0.      |
        +---------------------------+---------------+---------+
        | voroout                   | ``'-v'``      | 0.      |
        +---------------------------+---------------+---------+
        | meditview                 | ``'-g'``      | 0.      |
        +---------------------------+---------------+---------+
        | vtkview                   | ``'-k'``      | 0.      |
        +---------------------------+---------------+---------+
        | vtksurfview               | ``'-k'``      | 0.      |
        +---------------------------+---------------+---------+
        | nobound                   | ``'-B'``      | 0.      |
        +---------------------------+---------------+---------+
        | nonodewritten             | ``'-N'``      | 0.      |
        +---------------------------+---------------+---------+
        | noelewritten              | ``'-E'``      | 0.      |
        +---------------------------+---------------+---------+
        | nofacewritten             | ``'-F'``      | 0.      |
        +---------------------------+---------------+---------+
        | noiterationnum            | ``'-I'``      | 0.      |
        +---------------------------+---------------+---------+
        | nojettison                | ``'-J'``      | 0.      |
        +---------------------------+---------------+---------+
        | docheck                   | ``'-C'``      | 0.      |
        +---------------------------+---------------+---------+
        | quiet                     | ``'-Q'``      | 0.      |
        +---------------------------+---------------+---------+
        | nowarning                 | ``'-W'``      | 0.      |
        +---------------------------+---------------+---------+
        | verbose                   | ``'-V'``      | 0.      |
        +---------------------------+---------------+---------+

        Parameters of TetGen.

        +---------------------------+---------------+---------+
        | Option                    | Switch        | Default |
        +---------------------------+---------------+---------+
        | vertexperblock            | ``'-x'``      | 4092.   |
        +---------------------------+---------------+---------+
        | tetrahedraperblock        | ``'-x'``      | 8188.   |
        +---------------------------+---------------+---------+
        | shellfaceperblock         | ``'-x'``      | 2044.   |
        +---------------------------+---------------+---------+
        | supsteiner_level          | ``'-Y/'``     | 2.      |
        +---------------------------+---------------+---------+
        | addsteiner_algo           | ``'-Y//'``    | 1.      |
        +---------------------------+---------------+---------+
        | coarsen_param             | ``'-R'``      | 0.      |
        +---------------------------+---------------+---------+
        | weighted_param            | ``'-w'``      | 0.      |
        +---------------------------+---------------+---------+
        | opt_max_flip_level        | ``'-O'``      | 3.      |
        +---------------------------+---------------+---------+
        | opt_scheme                | ``'-O/#'``    | 7.      |
        +---------------------------+---------------+---------+
        | opt_iterations            | ``'-O//#'``   | 3.      |
        +---------------------------+---------------+---------+
        | smooth_cirterion          | ``'-s'``      | 1.      |
        +---------------------------+---------------+---------+
        | smooth_maxiter            | ``'-s'``      | 7.      |
        +---------------------------+---------------+---------+
        | order                     | ``'-o'``      | 1.      |
        +---------------------------+---------------+---------+
        | reversetetori             | ``'-o/'``     | 0.      |
        +---------------------------+---------------+---------+
        | steinerleft               | ``'-S'``      | 0.      |
        +---------------------------+---------------+---------+
        | unflip_queue_limit        | ``'-U#'``     | 1000.   |
        +---------------------------+---------------+---------+
        | hilbert_order             | ``'-b///'``   | 52.     |
        +---------------------------+---------------+---------+
        | hilbert_limit             | ``'-b//'``    |  8.     |
        +---------------------------+---------------+---------+
        | brio_threshold            | ``'-b'``      | 64.     |
        +---------------------------+---------------+---------+
        | brio_ratio                | ``'-b/'``     |0.125.   |
        +---------------------------+---------------+---------+
        | epsilon                   | ``'-T'``      | 1.0e-8. |
        +---------------------------+---------------+---------+
        | facet_separate_ang_tol    | ``'-p'``      | 179.9.  |
        +---------------------------+---------------+---------+
        | collinear_ang_tol         | ``'-p/'``     | 179.9.  |
        +---------------------------+---------------+---------+
        | facet_small_ang_tol       | ``'-p//'``    | 15.0.   |
        +---------------------------+---------------+---------+
        | maxvolume                 | ``'-a'``      | -1.0.   |
        +---------------------------+---------------+---------+
        | maxvolume_length          | ``'-a'``      | -1.0.   |
        +---------------------------+---------------+---------+
        | minratio                  | ``'-q'``      | 0.0.    |
        +---------------------------+---------------+---------+
        | mindihedral               | ``'-q'``      | 5.0.    |
        +---------------------------+---------------+---------+
        | optmaxdihedral            | ``'-o/#'``    | 177.0.  |
        +---------------------------+---------------+---------+
        | metric_scale              | ``'-m#'``     | 1.0.    |
        +---------------------------+---------------+---------+
        | smooth_alpha              | ``'-s'``      | 0.3.    |
        +---------------------------+---------------+---------+
        | coarsen_percent           | ``'-R1/#'``   | 1.0.    |
        +---------------------------+---------------+---------+
        | elem_growth_ratio         | ``'-r#'``     | 0.0.    |
        +---------------------------+---------------+---------+
        | refine_progress_ratio     | ``'-r/#'``    | 0.333.  |
        +---------------------------+---------------+---------+

        """

        # format switches
        if switches is None:
            switches_str = b""
        else:
            switches_str = bytes(switches, "utf-8")

        # check verbose switch
        if verbose == 0:
            quiet = 1

        # Call library
        plc = True  # always true
        try:
            self.node, self.elem = _tetgen.Tetrahedralize(
                self.v,
                self.f,
                switches_str,
                plc,
                psc,
                refine,
                quality,
                nobisect,
                cdt,
                cdtrefine,
                coarsen,
                weighted,
                brio_hilbert,
                flipinsert,
                metric,
                varvolume,
                fixedvolume,
                regionattrib,
                insertaddpoints,
                diagnose,
                convex,
                nomergefacet,
                nomergevertex,
                noexact,
                nostaticfilter,
                zeroindex,
                facesout,
                edgesout,
                neighout,
                voroout,
                meditview,
                vtkview,
                vtksurfview,
                nobound,
                nonodewritten,
                noelewritten,
                nofacewritten,
                noiterationnum,
                nojettison,
                docheck,
                quiet,
                nowarning,
                verbose,
                vertexperblock,
                tetrahedraperblock,
                shellfaceperblock,
                supsteiner_level,
                addsteiner_algo,
                coarsen_param,
                weighted_param,
                fliplinklevel,
                flipstarsize,
                fliplinklevelinc,
                opt_max_flip_level,
                opt_scheme,
                opt_iterations,
                smooth_cirterion,
                smooth_maxiter,
                delmaxfliplevel,
                order,
                reversetetori,
                steinerleft,
                unflip_queue_limit,
                no_sort,
                hilbert_order,
                hilbert_limit,
                brio_threshold,
                brio_ratio,
                epsilon,
                facet_separate_ang_tol,
                collinear_ang_tol,
                facet_small_ang_tol,
                maxvolume,
                maxvolume_length,
                minratio,
                opt_max_asp_ratio,
                opt_max_edge_ratio,
                mindihedral,
                optmaxdihedral,
                metric_scale,
                smooth_alpha,
                coarsen_percent,
                elem_growth_ratio,
                refine_progress_ratio,
            )
        except RuntimeError as e:
            raise RuntimeError(
                "Failed to tetrahedralize.\n"
                + "May need to repair surface by making it manifold:\n"
                # + str(e)
            )

        # check if a mesh was generated
        if not np.any(self.node):
            raise RuntimeError(
                "Failed to tetrahedralize.\n" + "May need to repair surface by making it manifold"
            )

        # Return nodes and elements
        self._updated = True

        return self.node, self.elem
