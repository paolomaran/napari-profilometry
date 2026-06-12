'''
Script for Phase Unwrapping via Max-flow/Min-cut (PUMA).
Reference: Bioucas-Dias and Valadão, IEEE TRANSACTIONS ON IMAGE PROCESSING, VOL. 16, NO. 3, MARCH 2007.

Author: Anik Ghosh, Politecnico di Milano
'''

# %%
import numpy as np
import maxflow 



def mincut(sourcesink, remain):
    """
    Python translation of the MATLAB MINCUT function.

    Parameters
    ----------
    sourcesink : ndarray (n x 3)
        Each row: [node, t_weight, s_weight]
        - t_weight = capacity (node -> source)
        - s_weight = capacity (node -> sink)

    remain : ndarray (m x 4)
        Each row: [u, v, cap_uv, cap_vu]

    Returns
    -------
    flow : float
        Maximum flow value

    cutside : ndarray (n x 2)
        [node, 0 or 1], where
        0 = node is on source side of the cut
        1 = node is on sink side of the cut
    """

    sourcesink = np.asarray(sourcesink, dtype=np.float32)

    # remain is a list of 2D arrays with potentially different row counts
    if len(remain) > 0:
        remain = np.vstack(remain).astype(np.float32)
    else:
        remain = np.zeros((0, 4), dtype=np.float32)

    # Number of nodes
    nodes = sourcesink[:, 0].astype(int)
    n_nodes = nodes.max() + 1  # must be consecutive indices

    # Create PyMaxflow graph
    g = maxflow.Graph[float]()
    node_ids = g.add_nodes(n_nodes)

    # Add terminal edges (source & sink capacities)
    for row in sourcesink:
        node, cap_to_source, cap_to_sink = row
        node = int(node)
        g.add_tedge(node_ids[node], cap_to_source, cap_to_sink)


    # Add bidirectional edges between nodes
    for row in remain:
        u, v, cap_uv, cap_vu = row
        u, v = int(u), int(v)
        g.add_edge(node_ids[u], node_ids[v], cap_uv, cap_vu)

    # Compute maxflow
    flow = g.maxflow()

    # Compute min-cut labels
    cutside = np.zeros((len(nodes), 2))
    for i, node in enumerate(nodes):
        cutside[i, 0] = node
        cutside[i, 1] = 0 if g.get_segment(node_ids[node]) == 0 else 1

    return float(flow), cutside


def clique_energy_ho(d, p, th, quant):
    """
    Python translation of MATLAB clique_energy_ho.m

    Computes clique energy:
        e = th^(p-2) * d^2       for d <= th
        e = d^p                  for d > th

    If quant == 'yes', values are quantized to multiples of 2π.

    Parameters
    ----------
    d : ndarray
        clique differences
    p : float
        power-law exponent
    th : float
        threshold separating quadratic & power-law regions
    quant : str ('yes' or 'no')
        whether to quantize differences

    Returns
    -------
    e : ndarray
        energy values
    """

    # Take absolute value or quantize first
    if quant == "no":
        d = np.abs(d)
    elif quant == "yes":
        d = np.abs(np.round(d / (2*np.pi)) * (2*np.pi))
    else:
        raise ValueError("quant must be 'yes' or 'no'")

    if th != 0:
        mask = d <= th
        e = (th**(p-2)) * d**2 * mask + d**p * (~mask)
    else:
        e = d**p

    return e



def energy_ho(kappa, psi, base, p, cliques, disc_bar, th, quant):
    """
    Python version of MATLAB: energy_ho.m

    Parameters
    ----------
    kappa : 2D ndarray
    psi : 2D ndarray
    base : 2D ndarray    (ROI mask padded outside)
    p : float            (clique exponent)
    cliques : ndarray shape (num_cliques, 2)
    disc_bar : 3D ndarray (m, n, num_cliques)
    th : float
    quant : str ('yes' or 'no')

    Returns
    -------
    erg : float
        Total energy value
    """

    m, n = psi.shape
    cliquesm, cliquesn = cliques.shape  # Size of input cliques
    maxdesl = int(np.max(np.abs(cliques)))   # Maximum clique length used

    H = 2 * maxdesl + 2 + m
    W = 2 * maxdesl + 2 + n

    base_kappa = np.zeros((H, W))
    base_kappa[maxdesl+1:maxdesl+1+m, maxdesl+1:maxdesl+1+n] = kappa

    psi_base = np.zeros((H, W))
    psi_base[maxdesl+1:maxdesl+1+m, maxdesl+1:maxdesl+1+n] = psi

    z = disc_bar.shape[2]

    # Initialize a zero array with shape (H, W, z)
    base_disc_bar = np.zeros((H, W, z))

    # Place disc_bar into the center, respecting padding
    base_disc_bar[maxdesl+1:maxdesl+1+m, maxdesl+1:maxdesl+1+n, :] = disc_bar

    # Preallocate arrays
    t_dkappa = np.zeros((H, W, cliquesm))
    a = np.zeros((H, W, cliquesm))

    # --- Compute clique differences ---
    for t in range(cliquesm):
        # Shift amounts (MATLAB uses [row, col])
        shift_r = int(cliques[t, 0])
        shift_c = int(cliques[t, 1])

        # circshift equivalent in NumPy
        auxili = np.roll(base_kappa, shift=(shift_r, shift_c), axis=(0, 1))
        t_dkappa[:, :, t] = base_kappa - auxili

        auxili2 = np.roll(psi_base, shift=(shift_r, shift_c), axis=(0, 1))
        dpsi = auxili2 - psi_base

        # circshift(base)
        base_shift = np.roll(base, shift=(shift_r, shift_c), axis=(0, 1))

        a[:, :, t] = (2 * np.pi * t_dkappa[:, :, t] - dpsi) \
                    * base * base_shift * base_disc_bar[:, :, t]

    # --- Compute energy ---
    # clique_energy_ho must operate elementwise
    energy = np.sum(clique_energy_ho(a, p, th, quant))

    return np.sum(energy)



def puma_ho(psi, p, *args):
    """
    Python port of the MATLAB puma_ho function.

    Parameters
    ----------
    psi : 2D numpy array (float)
        Wrapped phase image.
    p : float
        Clique potential exponent (>0).
    potential : dict, optional
        dictionary with keys 'quantized' ('yes'/'no') and 'threshold' (float).
        Defaults: {'quantized': 'no', 'threshold': np.pi}
    cliques : (K,2) array-like, optional
        Each row is a displacement vector [dr, dc]. Default [[1,0],[0,1]].
    qualitymaps : 3D numpy array (m,n,K), optional
        Quality map per clique. Values in [0,1]. Default zeros (no discontinuity).
    schedule : iterable of ints, optional
        Schedule of jump sizes. Default [1].
    verbose : 'yes' or 'no', optional
        If 'yes' some info would be displayed (kept but not plotted here).

    Returns
    -------
    unwph : 2D numpy array
        Unwrapped phase image.
    iter_count : int
        Number of iterations performed.
    erglist : list of floats
        Energy values per iteration.
    """

    # ----- Defaults -----
    potential = {}
    potential['quantized'] = 'no'
    potential['threshold'] = np.pi

    cliques = np.array([[1, 0],
                        [0, 1]])

    qualitymaps = np.zeros((psi.shape[0], psi.shape[1], 2))
    qual = 0

    schedule = [1]
    verbose = 'yes'

    if len([psi, p]) != 2:
        raise ValueError("Wrong number of required parameters")
    
    # Check optional arguments
    if len(args) % 2 == 1:
        raise ValueError("Optional parameters should always go by pairs")
    elif len(args) != 0:
        # Parse optional arguments
        for i in range(0, len(args), 2):
            key = args[i]
            value = args[i + 1]

            if key == 'potential':
                potential = value
            elif key == 'cliques':
                cliques = value
            elif key == 'qualitymaps':
                qualitymaps = value
                qual = 1
            elif key == 'schedule':
                schedule = value
            elif key == 'verbose':
                verbose = value
            else:
                raise ValueError(f"Unrecognized parameter: '{key}'")

    

    if qual == 1 and qualitymaps.shape[2] != cliques.shape[0]:
        raise ValueError(
            "qualitymaps must be a 3D matrix whose 3rd dimension is equal to the number of cliques. "
            "Each plane of qualitymaps corresponds to a clique."
        )



    th = potential['threshold']
    quant = potential['quantized']

    m, n = psi.shape  # Size of input
    kappa = np.zeros((m, n))  # Initial labeling
    # kappa = np.round(np.random.rand(m, n) * 40)  # Optional random initialization
    kappa_aux = kappa.copy()  # Copy of kappa
    iter = 0
    erglist = []  # List to store energies or metrics
    cliquesm, cliquesn = cliques.shape  # Size of input cliques
    
    if qual == 0:
        qualitymaps = np.zeros((psi.shape[0], psi.shape[1], cliques.shape[0]))

    disc_bar = 1 - qualitymaps

    maxdesl = int(np.max(np.abs(cliques)))

    # Create the larger zero-padded array
    base = np.zeros((2*maxdesl + 2 + m, 2*maxdesl + 2 + n))
    # Fill the central block with ones
    base[maxdesl+1 : maxdesl+1 + m, maxdesl+1 : maxdesl+1 + n] = 1
    

    # PROCESSING   %%%%%%%%%%%%%%%%%%%%%%%%

    for jump_size in schedule:
        possible_improvement = 1
        erg_previous = energy_ho(kappa, psi, base, p, cliques, disc_bar, th, quant)
        unwph = 2 * np.pi * kappa + psi
        while possible_improvement:
            iter += 1
            erglist.append(erg_previous)
            remain = []
            base_kappa = np.zeros((2*maxdesl + 2 + m, 2*maxdesl + 2 + n))
            base_kappa[maxdesl+1 : maxdesl+1 + m, maxdesl+1 : maxdesl+1 + n] = kappa

            psi_base = np.zeros((2*maxdesl + 2 + m, 2*maxdesl + 2 + n))
            psi_base[maxdesl+1 : maxdesl+1 + m, maxdesl+1 : maxdesl+1 + n] = psi

            z = disc_bar.shape[2]  # Number of 3rd dimension planes
            base_disc_bar = np.zeros((2*maxdesl + 2 + m, 2*maxdesl + 2 + n, z))
            base_disc_bar[maxdesl+1 : maxdesl+1 + m, maxdesl+1 : maxdesl+1 + n, :] = disc_bar

            # Preallocate arrays
            H = 2*maxdesl + 2 + m
            W = 2*maxdesl + 2 + n
            base_start = np.zeros((H, W, cliquesm))
            base_end   = np.zeros((H, W, cliquesm))
            source     = np.zeros((H, W, cliquesm))
            sink       = np.zeros((H, W, cliquesm))
            A          = np.zeros((H, W, cliquesm))
            B          = np.zeros((H, W, cliquesm))
            C          = np.zeros((H, W, cliquesm))
            D          = np.zeros((H, W, cliquesm))
            t_dkappa = np.zeros((H, W, cliquesm))
            a = np.zeros((H, W, cliquesm))

            for t in range(cliquesm):
                # Circularly shift the base array
                shift_neg = (-cliques[t, 0], -cliques[t, 1])
                shift_pos = (cliques[t, 0], cliques[t, 1])                
                base_start[:, :, t] = np.roll(base, shift_neg, axis=(0, 1)) * base
                base_end[:, :, t] = np.roll(base, shift_pos, axis=(0, 1)) * base

                # Circularly shift the arrays
                auxili = np.roll(base_kappa, (cliques[t, 0], cliques[t, 1]), axis=(0, 1))
                t_dkappa[:, :, t] = base_kappa - auxili

                auxili2 = np.roll(psi_base, (cliques[t, 0], cliques[t, 1]), axis=(0, 1))
                dpsi = auxili2 - psi_base

                # Compute shifted base
                shifted_base = np.roll(base, (cliques[t, 0], cliques[t, 1]), axis=(0, 1))

                # Compute a
                a[:, :, t] = (2 * np.pi * t_dkappa[:, :, t] - dpsi) * base * shifted_base

                # Compute A, D, C, B using clique_energy_ho
                A[:, :, t] = clique_energy_ho(np.abs(a[:, :, t]), p, th, quant) * base * shifted_base * base_disc_bar[:, :, t]
                D[:, :, t] = A[:, :, t]
                C[:, :, t] = clique_energy_ho(np.abs(2 * np.pi * jump_size + a[:, :, t]), p, th, quant) * base * shifted_base * base_disc_bar[:, :, t]
                B[:, :, t] = clique_energy_ho(np.abs(-2 * np.pi * jump_size + a[:, :, t]), p, th, quant) * base * shifted_base * base_disc_bar[:, :, t]

                # Compute shifted differences and clip negatives to zero
                source[:, :, t] = np.roll(np.maximum(C[:, :, t] - A[:, :, t], 0), (-cliques[t, 0], -cliques[t, 1]), axis=(0, 1)) * base_start[:, :, t]
                sink[:, :, t]   = np.roll(np.maximum(A[:, :, t] - C[:, :, t], 0), (-cliques[t, 0], -cliques[t, 1]), axis=(0, 1)) * base_start[:, :, t]

                # Add contributions from end positions
                source[:, :, t] += np.maximum(D[:, :, t] - C[:, :, t], 0) * base_end[:, :, t]
                sink[:, :, t]   += np.maximum(C[:, :, t] - D[:, :, t], 0) * base_end[:, :, t]

            # Remove border of size maxdesl+1 from all sides
            slice_rows = slice(maxdesl + 1, maxdesl + 1 + m)
            slice_cols = slice(maxdesl + 1, maxdesl + 1 + n)

            source = source[slice_rows, slice_cols, :]
            sink = sink[slice_rows, slice_cols, :]
            auxiliar1 = (B + C - A - D)[slice_rows, slice_cols, :]
            base_start = base_start[slice_rows, slice_cols, :]
            base_end = base_end[slice_rows, slice_cols, :]

            for t in range(cliquesm):
                start = np.flatnonzero(base_start[:, :, t] != 0)
                endd = np.flatnonzero(base_end[:, :, t] != 0)
                auxiliar2 = auxiliar1[:, :, t]
                
                # Compute clipped values of auxiliar2 at endd indices
                values = np.maximum(auxiliar2.flatten()[endd], 0)
                
                # Create auxiliar3 array with 4 columns
                auxiliar3 = np.column_stack((start, endd, values, np.zeros_like(endd)))
                
                # Append to remain
                remain.append(auxiliar3)

            sourcefinal = np.sum(source, axis=2)
            sinkfinal = np.sum(sink, axis=2)
            # Create sourcesink array with indices and flattened sums
            indices = np.arange(m*n)  # MATLAB uses 1-based indexing
            sourcesink = np.column_stack((indices, sourcefinal.flatten(), sinkfinal.flatten()))

            # KAPPA RELABELING
            flow, cutside = mincut(sourcesink, remain)  

            # Adjust for 0-based indexing if needed

            idx = cutside[:, 0].astype(int)
            kappa_aux_flat = kappa_aux.flatten()
            kappa_flat = kappa.flatten()

            kappa_flat[idx] = kappa_flat[idx] + (1 - cutside[:,1]) * jump_size

            # Reshape back if necessary
            #kappa_aux = kappa_flat.reshape(kappa_aux.shape)
            kappa_aux = kappa_flat.reshape(m, n)

            # kappa_aux[cutside[:, 0].astype(int)] = kappa[cutside[:, 0].astype(int)] + (1 - cutside[:, 1]) * jump_size
            # CHECK ENERGY IMPROVEMENT
            erg_actual = energy_ho(kappa_aux, psi, base, p, cliques, disc_bar, th, quant)
            
            if erg_actual < erg_previous:
                erg_previous = erg_actual
                kappa = kappa_aux.copy()
            else:
                possible_improvement = 0
                unwph = 2 * np.pi * kappa + psi

            if verbose == 'yes':
                # Optionally display the current unwrapped phase
                # import matplotlib.pyplot as plt
                # plt.imshow(2 * np.pi * kappa + psi, cmap='gray')
                # plt.draw()
                # plt.pause(0.01)  # small pause to update the figure
            
            # Delete variables to free memory
                del base_start, base_end, source, sink, auxiliar1, auxiliar2, A, B, C, D

    return unwph, iter, erglist


def funcPUMA(eta):
    """
    eta      : wrapped phase image (numpy array)
    puma_ho  : function handle to PUMA-HO algorithm in Python
    Returns:
        unwph : unwrapped phase image
    """

    p = 0.5  # clique potential exponent
    potential = {}
    potential['quantized'] = 'no'
    potential['threshold'] = 0

    # Clique directions (same as MATLAB: [1 0; 0 1])
    cliques = np.array([[1, 0],
                    [0, 1]])

    # Create a new figure
    # plt.figure()

    # Call the Python version of puma_ho
    unwph, _, _ = puma_ho(eta, p, 'potential', potential, 'cliques', cliques)
    
    return unwph


