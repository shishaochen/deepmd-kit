import numpy as np

from typing import Dict, List, Tuple

from deepmd.descriptor import Descriptor
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION, op_module, tf


@Descriptor.register("nb")
class DescrptNb(Descriptor):
    """This descriptor tells indexes and types of neighbor atoms sorted by distance.

    Parameters
    ----------
    rcut: float
            The cut-off radius.
    sel: list[str]
            sel[i] specifies the maxmum number of type i atoms in the cut-off radius.
    """

    def __init__(self, rcut: float, sel: List[str]) -> None:
        self.rcut = rcut
        self.sel = sel
        self.ntypes = len(self.sel)
        self.nnei = np.cumsum(self.sel)[-1]
        self.ndescrpt = self.nnei * 3

    def get_rcut(self) -> float:
        return self.rcut

    def get_ntypes(self) -> int:
        return self.ntypes

    def get_dim_out(self) -> int:
        """Length of descriptor for each atom to neighbor element.
        The descriptor shape should be [nframes*natoms[0], self.get_dim_out()].
        """
        return self.ndescrpt

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
            _ = tf.constant(self.rcut, name='rcut', dtype=GLOBAL_TF_FLOAT_PRECISION)
            _ = tf.constant(self.ntypes, name='ntypes', dtype=tf.int32)
            _ = tf.constant(self.ndescrpt, name='ndescrpt', dtype=tf.int32)
            _ = tf.constant(self.sel, name='sel', dtype=tf.int32)

        coord = tf.reshape(coord_, [-1, natoms[1] * 3])
        box = tf.reshape(box_, [-1, 9])
        atype = tf.reshape(atype_, [-1, natoms[1]])

        self.rij, self.nlist \
            = op_module.prod_nb_mat(coord,
                                    atype,
                                    natoms,
                                    box,
                                    mesh,
                                    rcut=self.rcut,
                                    sel=self.sel)

        tf.summary.histogram('rij', self.rij)
        self.descrpt_out = tf.reshape(self.rij, [-1, self.get_dim_out()])
        return self.descrpt_out

    def prod_force_virial(self,
                          atom_ener: tf.Tensor,
                          natoms: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        [net_deriv] = tf.gradients(atom_ener, self.descrpt_out)
        tf.summary.histogram('net_derivative', net_deriv)
        net_deriv_reshape = tf.reshape(net_deriv, [-1, natoms[0] * self.ndescrpt])        
        force = op_module.prod_force_nb(net_deriv_reshape,
                                        self.nlist,
                                        natoms)
        virial, atom_virial = op_module.prod_virial_nb(net_deriv_reshape,
                                                       self.rij,
                                                       self.nlist,
                                                       natoms)
        tf.summary.histogram('force', force)
        tf.summary.histogram('virial', virial)
        tf.summary.histogram('atom_virial', atom_virial)
        return force, virial, atom_virial
