"""
    DO jb = i_startblk, i_endblk
    CALL get_indices_e(p_patch, jb, i_startblk, i_endblk, i_startidx, i_endidx, 7, -9)
    DO jk = 1, nlev
        DO je = i_startidx, i_endidx
        z_v_grad_w(je, jk, jb) = p_diag % vn_ie(je, jk, jb) * p_patch % edges % inv_dual_edge_length(je, jb) * (p_prog % w(p_patch % edges % cell_idx(je, jb, 1), jk, p_patch % edges % cell_blk(je, jb, 1)) - p_prog % w(p_patch % edges % cell_idx(je, jb, 2), jk, p_patch % edges % cell_blk(je, jb, 2))) + z_vt_ie(je, jk, jb) * p_patch % edges % inv_primal_edge_length(je, jb) * p_patch % edges % tangent_orientation(je, jb) * (z_w_v(p_patch % edges % vertex_idx(je, jb, 1), jk, p_patch % edges % vertex_blk(je, jb, 1)) - z_w_v(p_patch % edges % vertex_idx(je, jb, 2), jk, p_patch % edges % vertex_blk(je, jb, 2)))
        END DO
    END DO
    END DO
"""

