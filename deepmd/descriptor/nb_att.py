from typing import List

from deepmd.descriptor import Descriptor
from deepmd.descriptor.nb import DescrptNb
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION, tf


@Descriptor.register("nb_att")
class DescrptNbAtt(DescrptNb):

    def __init__(self, rcut: float, sel: List[str], qk_units: int, v_units: int) -> None:
        _ = super(DescrptNbAtt, self).__init__(rcut, sel)
        self.qk_units = qk_units
        self.v_units = v_units

    def get_dim_out(self) -> int:
        return self.nnei*self.v_units

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
        _ = super(DescrptNbAtt, self).build(
            coord_, atype_, natoms, box_, mesh, input_dict, reuse, suffix)

        raw_rij = tf.reshape(self.rij, [-1, self.nnei, 3])  # 邻居间 3 个轴向坐标差
        radius = tf.norm(raw_rij, axis=-1, keepdims=True)  # 邻居间径向距离 shape=[batch*nloc,nnei,1]
        rel_xyz = raw_rij / radius  # 邻居间 3 个轴向坐标差比上径向距离 shape=[batch*nloc,nnei,3]
        atype = tf.reshape(atype_, [-1])  # shape=[batch*nall]
        return self._pass_filter(atype, radius, rel_xyz, reuse)

    def _pass_filter(self,
                     atype: tf.Tensor,  # shape=[batch*nall]
                     radius: tf.Tensor,  # shape=[batch*nloc,nnei,1]
                     rel_xyz: tf.Tensor,  # shape=[batch*nloc,nnei,3]
                     reuse: bool):
        with tf.variable_scope('filter_attention', reuse=reuse):
            self.q_mat = [None]*self.ntypes
            self.k_mat = [[None]*self.ntypes]*self.ntypes
            self.v_mat = [[None]*self.ntypes]*self.ntypes
            for type_i in range(self.ntypes):  # left neightbor
                self.q_mat[type_i] = tf.get_variable(
                    shape=[self.ntypes, 3, self.qk_units],
                    dtype=GLOBAL_TF_FLOAT_PRECISION,
                    name=f'q_mat_{type_i}')
                for type_j in range(self.ntypes):  # right neightbor
                    self.k_mat[type_i][type_j] = tf.get_variable(
                        shape=[self.ntypes, 3, self.qk_units],
                        dtype=GLOBAL_TF_FLOAT_PRECISION,
                        name=f'k_mat_{type_i}_{type_j}')
                    self.v_mat[type_i][type_j] = tf.get_variable(
                        shape=[self.ntypes, 3, self.v_units],
                        dtype=GLOBAL_TF_FLOAT_PRECISION,
                        name=f'v_mat_{type_i}_{type_j}')

        # Embedding net with attention structure
        start_index_i = 0
        qkv_list = []
        for type_i in range(self.ntypes):
            nei_type_i = self.sel[type_i]
            rad_i = tf.slice(radius, [0, start_index_i, 0], [-1, nei_type_i, 1])
            xyz_i = tf.slice(rel_xyz, [0, start_index_i, 0], [-1, nei_type_i, 3])
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
                nei_type_j = self.sel[type_j]
                rad_j = tf.slice(radius, [0, start_index_j, 0], [-1, nei_type_j, 1])
                xyz_j = tf.slice(rel_xyz, [0, start_index_j, 0], [-1, nei_type_j, 3])
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
