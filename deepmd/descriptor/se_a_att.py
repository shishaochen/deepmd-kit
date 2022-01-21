from matplotlib.pyplot import axis
import numpy as np

from typing import Dict, List, Tuple

from deepmd.descriptor import Descriptor
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION, op_module, tf


@Descriptor.register("se_a_att")
class DescrptSeAAtt(Descriptor):

    def __init__(self, rcut: float, sel: List[str], qk_units: int, v_units: int) -> None:
        self.rcut_r = rcut
        self.rcut_a = -1
        self.sel_a = sel
        self.sel_r = [0]*len(self.sel_a)
        self.ntypes = len(self.sel_a)
        self.nnei_a = np.sum(self.sel_a)
        self.nnei_r = 0
        self.ndescrpt = self.nnei_a * 4
        self.qk_units = qk_units
        self.v_units = v_units

    def get_rcut(self) -> float:
        return self.rcut_r

    def get_ntypes(self) -> int:
        return self.ntypes

    def get_dim_out(self) -> int:
        return self.nnei_a*self.v_units

    def get_tensor_names(self, suffix: str = "") -> Tuple[str]:
        return tuple()

    def pass_tensors_from_frz_model(self, *tensors: tf.Tensor) -> None:
        pass

    def init_variables(self, model_file: str, suffix: str= "") -> None:
        pass

    def compute_input_stats(self,
                            data_coord: List[np.ndarray],
                            data_box: List[np.ndarray],
                            data_atype: List[np.ndarray],
                            natoms_vec: List[np.ndarray],
                            mesh: List[np.ndarray],
                            input_dict: Dict[str, List[np.ndarray]]
                            ) -> None:
        pass

    def build(self,
              coord_: tf.Tensor,
              atype_: tf.Tensor,
              natoms: tf.Tensor,
              box_: tf.Tensor,
              mesh: tf.Tensor,
              input_dict: dict,
              reuse: bool = None,
              suffix: str = ''
    ) -> tf.Tensor:
        with tf.variable_scope('descrpt_attr' + suffix, reuse=reuse):
            _ = tf.constant(self.rcut_r, name='rcut', dtype=GLOBAL_TF_FLOAT_PRECISION)
            _ = tf.constant(self.ntypes, name='ntypes', dtype=tf.int32)
            _ = tf.constant(self.ndescrpt, name='ndescrpt', dtype=tf.int32)
            _ = tf.constant(self.sel_a, name='sel', dtype=tf.int32)
            t_avg = tf.zeros(shape=[self.ntypes, self.ndescrpt], 
                             dtype=GLOBAL_TF_FLOAT_PRECISION,
                             name='t_avg')
            t_std = tf.ones(shape=[self.ntypes, self.ndescrpt],
                             dtype=GLOBAL_TF_FLOAT_PRECISION,
                             name='t_std')

        coord = tf.reshape(coord_, [-1, natoms[1] * 3])
        box = tf.reshape(box_, [-1, 9])
        atype = tf.reshape(atype_, [-1, natoms[1]])
        self.descrpt, self.descrpt_deriv, self.rij, self.nlist \
            = op_module.prod_env_mat_a(coord,
                                       atype,
                                       natoms,
                                       box,
                                       mesh,
                                       t_avg,
                                       t_std,
                                       rcut_a=self.rcut_a,
                                       rcut_r=self.rcut_r,
                                       rcut_r_smth=self.rcut_r,  # No smoothing
                                       sel_a=self.sel_a,
                                       sel_r=self.sel_r)
        tf.summary.histogram('descrpt', self.descrpt)
        tf.summary.histogram('rij', self.rij)
        tf.summary.histogram('nlist', self.nlist)

        atype = tf.reshape(atype_, [-1])  # shape=[batch*nall]
        descrpt_reshape = tf.reshape(self.descrpt, [-1, self.nnei_a, 4])  # shape=[batch*nloc,self.nnei_a,4]
        nlist = tf.reshape(self.nlist, [-1, self.nnei_a])  # shape=[batch*nloc,self.nnei_a]
        return self._pass_filter(atype, descrpt_reshape, nlist, reuse)

    def prod_force_virial(self,
                          atom_ener: tf.Tensor,
                          natoms: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        net_deriv = tf.gradients(atom_ener, self.descrpt)
        net_deriv_reshape = tf.reshape(net_deriv, [-1, natoms[0]*self.ndescrpt])
        force = op_module.prod_force_se_a(net_deriv_reshape,
                                          self.descrpt_deriv,
                                          self.nlist,
                                          natoms,
                                          n_a_sel=self.nnei_a,
                                          n_r_sel=self.nnei_r)
        virial, atom_virial = op_module.prod_virial_se_a(net_deriv_reshape,
                                                         self.descrpt_deriv,
                                                         self.rij,
                                                         self.nlist,
                                                         natoms,
                                                         n_a_sel=self.nnei_a,
                                                         n_r_sel=self.nnei_r)
        return force, virial, atom_virial

    def _pass_filter(self,
                     atype: tf.Tensor,  # shape=[batch*nall]
                     descrpt: tf.Tensor,  # shape=[batch*nloc,self.nnei_a,4]
                     nlist: tf.Tensor,  # shape=[batch*nloc,self.nnei_a]
                     reuse: bool):
        with tf.variable_scope('filter_attention', reuse=reuse):
            self.q_mat = [None]*self.ntypes
            self.k_mat = [[None]*self.ntypes]*self.ntypes
            self.v_mat = [[None]*self.ntypes]*self.ntypes
            for type_i in range(self.ntypes):  # left neighbor
                self.q_mat[type_i] = tf.get_variable(
                    shape=[self.ntypes, 3, self.qk_units],
                    dtype=GLOBAL_TF_FLOAT_PRECISION,
                    name=f'q_mat_{type_i}')
                for type_j in range(self.ntypes):  # right neighbor
                    self.k_mat[type_i][type_j] = tf.get_variable(
                        shape=[self.ntypes, 3, self.qk_units],
                        dtype=GLOBAL_TF_FLOAT_PRECISION,
                        name=f'k_mat_{type_i}_{type_j}')
                    self.v_mat[type_i][type_j] = tf.get_variable(
                        shape=[self.ntypes, 3, self.v_units],
                        dtype=GLOBAL_TF_FLOAT_PRECISION,
                        name=f'v_mat_{type_i}_{type_j}')

        # Convert from [1/r, x/r^2, y/r^2, z/r^2] to [r, x/r, y/r, z/r]
        inv_rad = tf.slice(descrpt, [0, 0, 0], [-1, -1, 1])  # [1/r] while non-neighbor one is 0.
        rel_xyz = tf.slice(descrpt, [0, 0, 1], [-1, -1, 3])  # [x/r^2, y/r^2, z/r^2] while non-neighbor one is 0.
        padded = tf.expand_dims(nlist < 0, axis=-1)
        ones = tf.ones_like(inv_rad, dtype=descrpt.dtype)
        coeff = 1. / tf.where(padded, ones, inv_rad)  # [r] while non-neighbor one is 1.
        radius = inv_rad * coeff * coeff  # [r] while non-neighbor one is 0.
        rel_xyz = rel_xyz * coeff  # [x/r, y/r, z/r] while non-neighbor one is 0.

        # Embedding net with attention structure
        start_index_i = 0
        qkv_list = []
        for type_i in range(self.ntypes):
            nei_type_i = self.sel_a[type_i]
            rad_i = tf.slice(radius, [0, start_index_i, 0], [-1, nei_type_i, -1])
            xyz_i = tf.slice(rel_xyz, [0, start_index_i, 0], [-1, nei_type_i, -1])
            start_index_j = start_index_i
            start_index_i += nei_type_i

            # Generate Q from center + neighbor_i
            zeros = tf.zeros_like(rad_i)  # shape=[batch*loc,nei_type_i,1]
            fea_q = tf.concat([rad_i, zeros, rad_i], axis=-1)  # shape=[batch*loc,nei_type_i,3]
            q_param = tf.gather(self.q_mat[type_i], indices=atype, axis=0)  # shape=[batch*nall,3,qk_units]
            q_val = tf.matmul(fea_q, q_param)  # shape=[batch*nall,nei_type_i,qk_units]
            q_val = tf.reshape(q_val, [-1, self.qk_units])  # shape=[batch*nall*nei_type_i,qk_units]

            # Generate K & V from enter + neighbor_i + neighbor_j
            k_list = []
            v_list = []
            for type_j in range(type_i, self.ntypes):  # Avoid duplicated angles
                nei_type_j = self.sel_a[type_j]
                rad_j = tf.slice(radius, [0, start_index_j, 0], [-1, nei_type_j, -1])
                xyz_j = tf.slice(rel_xyz, [0, start_index_j, 0], [-1, nei_type_j, -1])
                start_index_j += nei_type_j

                ii = tf.repeat(rad_i, nei_type_j, axis=1)  # shape=[batch*nloc,nei_type_i*nei_type_j, 1]
                jj = tf.tile(rad_j, [1, nei_type_i, 1])  # shape=[batch*nloc,nei_type_i*nei_type_j, 1]
                theta = tf.einsum('bim,bjm->bij', xyz_i, xyz_j)  # shape=[batch*nloc,nei_type_i,nei_type_j]
                tt = tf.reshape(theta, [-1, nei_type_i*nei_type_j, 1])  # shape=[batch*nloc,nei_type_i*nei_type_j, 1]
                fea_ij = tf.concat([ii, tt, jj], axis=-1)  # shape=[batch*nloc,nei_type_i*nei_type_j, 3]
                k_param = tf.gather(self.k_mat[type_i][type_j], indices=atype, axis=0)  # shape=[batch*nall,3,qk_units]
                k_val = tf.matmul(fea_ij, k_param)  # shape=[batch*nloc,nei_type_i*nei_type_j,qk_units]
                k_val = tf.reshape(k_val, [-1, nei_type_j, self.qk_units])  # shape=[batch*nloc*nei_type_i,nei_type_j,qk_units]
                v_param = tf.gather(self.v_mat[type_i][type_j], indices=atype, axis=0)  # shape=[batch*nall,3,v_units]
                v_val = tf.matmul(fea_ij, v_param)  # shape=[batch*nloc,nei_type_i*nei_type_j,v_units]
                v_val = tf.reshape(v_val, [-1, nei_type_j, self.v_units])  # shape=[batch*nloc*nei_type_i,nei_type_j,v_units]

                k_list.append(k_val)
                v_list.append(v_val)
            k_val = tf.concat(k_list, axis=-2)  # shape=[batch*nloc*nei_type_i,nnei,qk_units]
            v_val = tf.concat(v_list, axis=-2)  # shape=[batch*nloc*nei_type_i,nnei,v_units]

            # Calculate combination of Q & K & V
            qk_val = tf.einsum('bm,bjm->bj', q_val, k_val)  # shape=[batch*nloc*nei_type_i,nnei]
            per_i_sum = tf.reduce_sum(tf.abs(qk_val), axis=-1, keepdims=True) + 1e-8  # shape=[batch*nloc*nei_type_i,1]
            scaled = qk_val / per_i_sum  # shape=[batch*nloc*nei_type_i,nnei]
            qkv = tf.einsum('bj,bjm->bm', scaled, v_val)  # shape=[batch*nloc*nei_type_i,v_units]
            qkv = tf.reshape(qkv, [-1, nei_type_i*self.v_units])  # shape=[batch*nloc,nei_type_i*v_units]
            qkv_list.append(qkv)
        return tf.concat(qkv_list, axis=-1)  # shape=[batch*nloc,nnei*v_units]
