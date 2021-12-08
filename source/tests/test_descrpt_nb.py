import numpy as np
import os
import unittest

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from deepmd.common import data_requirement
from deepmd.env import GLOBAL_NP_FLOAT_PRECISION, op_module
from deepmd.utils import random as dp_random
from deepmd.utils.data_system import DeepmdDataSystem


CUR_DIR = os.path.dirname(__file__)
kDataSystems = [
    os.path.join(CUR_DIR, '../../examples', 'water/data/data_0'),
    os.path.join(CUR_DIR, '../../examples', 'water/data/data_1'),
    os.path.join(CUR_DIR, '../../examples', 'water/data/data_2')
]


def base_se_a(rcut, rcut_smth, sel, batch, mean, stddev):
    g = tf.Graph()
    with g.as_default():
        coord = tf.placeholder(GLOBAL_NP_FLOAT_PRECISION, [None, None])
        box = tf.placeholder(GLOBAL_NP_FLOAT_PRECISION, [None, None])
        atype = tf.placeholder(tf.int32, [None, None])
        natoms_vec = tf.placeholder(tf.int32, [None])
        default_mesh = tf.placeholder(tf.int32, [None])
        _, _, rij, nlist = op_module.prod_env_mat_a(coord,
                                                    atype,
                                                    natoms_vec,
                                                    box,
                                                    default_mesh,
                                                    tf.constant(mean),
                                                    tf.constant(stddev),
                                                    rcut_a=-1.,
                                                    rcut_r=rcut,
                                                    rcut_r_smth=rcut_smth,
                                                    sel_a=sel,
                                                    sel_r=[0, 0])

    with tf.Session(graph=g) as sess:
        return sess.run([rij, nlist], feed_dict={
            coord: batch['coord'],
            box: batch['box'],
            natoms_vec: batch['natoms_vec'],
            atype: batch['type'],
            default_mesh: np.array([0, 0, 0, 2, 2, 2])
        })


def peer_nb(rcut, sel, batch):
    g = tf.Graph()
    with g.as_default():
        coord = tf.placeholder(GLOBAL_NP_FLOAT_PRECISION, [None, None])
        box = tf.placeholder(GLOBAL_NP_FLOAT_PRECISION, [None, None])
        atype = tf.placeholder(tf.int32, [None, None])
        natoms_vec = tf.placeholder(tf.int32, [None])
        default_mesh = tf.placeholder(tf.int32, [None])
        t_rij, t_nlist = op_module.prod_nb_mat(coord,
                                               atype,
                                               natoms_vec,
                                               box,
                                               default_mesh,
                                               rcut=rcut,
                                               sel=sel)

    with tf.Session(graph=g) as sess:
        return sess.run([t_rij, t_nlist], feed_dict={
            coord: batch['coord'],
            box: batch['box'],
            natoms_vec: batch['natoms_vec'],
            atype: batch['type'],
            default_mesh: np.array([0, 0, 0, 2, 2, 2])
        })


def forward_nb(rcut, sel, batch):
    g = tf.Graph()
    with g.as_default():
        coord = tf.placeholder(GLOBAL_NP_FLOAT_PRECISION, [None, None])
        box = tf.placeholder(GLOBAL_NP_FLOAT_PRECISION, [None, None])
        atype = tf.placeholder(tf.int32, [None, None])
        natoms_vec = tf.placeholder(tf.int32, [None])
        default_mesh = tf.placeholder(tf.int32, [None])
        net_deriv = tf.placeholder(GLOBAL_NP_FLOAT_PRECISION, [None, None])
        rij, nlist = op_module.prod_nb_mat(coord,
                                           atype,
                                           natoms_vec,
                                           box,
                                           default_mesh,
                                           rcut=rcut,
                                           sel=sel)
        force = op_module.prod_force_nb(net_deriv, nlist, natoms_vec)
        virial, atom_virial = op_module.prod_virial_nb(net_deriv, rij, nlist, natoms_vec)

    with tf.Session(graph=g) as sess:
        return sess.run([rij, nlist, force, virial, atom_virial], feed_dict={
            coord: batch['coord'],
            box: batch['box'],
            natoms_vec: batch['natoms_vec'],
            atype: batch['type'],
            net_deriv: batch['net_deriv'],
            default_mesh: np.array([0, 0, 0, 2, 2, 2])
        })


def calc_force(net_deriv, nlist, nloc, nall, nnei):
    nsamples = net_deriv.shape[0]
    ndescrpt = nnei*3
    force = np.zeros(shape=[nsamples, nall*3], dtype=GLOBAL_NP_FLOAT_PRECISION)

    # Force from neighbors
    for ss in range(nsamples):
        for aa in range(nloc):
            force_offset = aa*3
            for nn in range(nnei):
                deriv_offset = aa*ndescrpt + nn*3
                force[ss, force_offset+0] += net_deriv[ss, deriv_offset+0]
                force[ss, force_offset+1] += net_deriv[ss, deriv_offset+1]
                force[ss, force_offset+2] += net_deriv[ss, deriv_offset+2]

    # Force to neighbors
    for ss in range(nsamples):
        for aa in range(nloc):
            for nn in range(nnei):
                nid = nlist[ss, aa*nnei+nn]
                if nid < 0: continue
                force_offset = nid*3
                deriv_offset = aa*ndescrpt + nn*3
                force[ss, force_offset+0] -= net_deriv[ss, deriv_offset+0]
                force[ss, force_offset+1] -= net_deriv[ss, deriv_offset+1]
                force[ss, force_offset+2] -= net_deriv[ss, deriv_offset+2]

    return force


def calc_virial(net_deriv, rij, nlist, nloc, nall, nnei):
    nsamples = net_deriv.shape[0]
    ndescrpt = nnei*3
    virial = np.zeros(shape=[nsamples, 9], dtype=GLOBAL_NP_FLOAT_PRECISION)
    atom_virial = np.zeros(shape=[nsamples, nall*9], dtype=GLOBAL_NP_FLOAT_PRECISION)
    for ss in range(nsamples):
        for aa in range(nloc):
            for nn in range(nnei):
                nid = nlist[ss, aa*nnei+nn]
                if nid < 0: continue
                deriv_offset = aa*ndescrpt + nn*3
                for di in range(3):
                    ff = net_deriv[ss, deriv_offset+di]
                    for dj in range(3):
                        vv = ff * rij[ss, aa*ndescrpt+nn*3+dj]
                        virial[ss, di*3+dj] -= vv
                        atom_virial[ss, nid*9+di*3+dj] -= vv
    return virial, atom_virial


def backward_nb(rcut, sel, batch):
    g = tf.Graph()
    with g.as_default():
        coord = tf.placeholder(GLOBAL_NP_FLOAT_PRECISION, [None, None])
        box = tf.placeholder(GLOBAL_NP_FLOAT_PRECISION, [None, None])
        atype = tf.placeholder(tf.int32, [None, None])
        natoms_vec = tf.placeholder(tf.int32, [None])
        default_mesh = tf.placeholder(tf.int32, [None])
        force_grad = tf.placeholder(GLOBAL_NP_FLOAT_PRECISION, [None, None])
        virial_grad = tf.placeholder(GLOBAL_NP_FLOAT_PRECISION, [None, None])
        rij, nlist = op_module.prod_nb_mat(coord,
                                           atype,
                                           natoms_vec,
                                           box,
                                           default_mesh,
                                           rcut=rcut,
                                           sel=sel)
        grad_from_force = op_module.prod_force_nb_grad(force_grad, nlist, natoms_vec)
        grad_from_virial = op_module.prod_virial_nb_grad(virial_grad, rij, nlist, natoms_vec)

    with tf.Session(graph=g) as sess:
        return sess.run([rij, nlist, grad_from_force, grad_from_virial], feed_dict={
            coord: batch['coord'],
            box: batch['box'],
            natoms_vec: batch['natoms_vec'],
            atype: batch['type'],
            force_grad: batch['force_grad'],
            virial_grad: batch['virial_grad'],
            default_mesh: np.array([0, 0, 0, 2, 2, 2])
        })


def calc_grad_from_force(force_grad, nlist, nloc, nnei):
    nsamples = force_grad.shape[0]
    ndescrpt = nnei*3
    grad_out = np.zeros(shape=[nsamples, nloc*ndescrpt], dtype=GLOBAL_NP_FLOAT_PRECISION)

    # Force from neighbors
    for ss in range(nsamples):
        for aa in range(nloc):
            for nn in range(nnei):
                for dd in range(3):
                    grad_out[ss, aa*ndescrpt+nn*3+dd] += force_grad[ss, aa*3+dd]

    # Force to neighbors
    for ss in range(nsamples):
        for aa in range(nloc):
            for nn in range(nnei):
                nid = nlist[ss, aa*nnei+nn]
                if nid > nloc: nid = nid % nloc
                if nid < 0: continue
                for dd in range(3):
                    grad_out[ss, aa*ndescrpt+nn*3+dd] -= force_grad[ss, nid*3+dd]

    return grad_out


def calc_grad_from_virial(virial_grad, rij, nlist, nloc, nnei):
    nsamples = virial_grad.shape[0]
    ndescrpt = nnei*3
    grad_out = np.zeros(shape=[nsamples, nloc*ndescrpt], dtype=GLOBAL_NP_FLOAT_PRECISION)
    for ss in range(nsamples):
        for aa in range(nloc):
            for nn in range(nnei):
                nid = nlist[ss, aa*nnei+nn]
                if nid > nloc: nid = nid % nloc
                if nid < 0: continue
                for di in range(3):
                    for dj in range(3):
                        grad_out[ss, aa*ndescrpt+nn*3+di] -= virial_grad[ss, di*3+dj] * rij[ss, aa*ndescrpt+nn*3+dj]
    return grad_out


class TestNb(unittest.TestCase):

    def setUp(self):
        self.rcut = 6.
        self.rcut_smth = 0.5
        self.sel = [46, 92]
        self.sec = np.cumsum(self.sel)
        self.nnei = sum(self.sel)
        self.type_map = ['O', 'H']
        self.ntypes = len(self.type_map)
        self.nloc = 192
        self.nall = 192

        self.batch_size = 3
        dp_random.seed(7)
        self.dataset = DeepmdDataSystem(
            systems=kDataSystems,
            batch_size=self.batch_size,
            test_size=1,
            rcut=self.rcut,
            type_map=self.type_map,
            trn_all_set=True
        )
        self.dataset.add_dict(data_requirement)

    def test_consistency(self):
        batch = self.dataset.get_batch()
        avg_zero = np.zeros([self.ntypes, self.nnei*4]).astype(GLOBAL_NP_FLOAT_PRECISION)
        std_ones = np.ones([self.ntypes, self.nnei*4]).astype(GLOBAL_NP_FLOAT_PRECISION)
        base_rij, base_nlist = base_se_a(
            rcut=self.rcut,
            rcut_smth=self.rcut_smth,
            sel=self.sel,
            batch=batch,
            mean=avg_zero,
            stddev=std_ones
        )
        peer_rij, peer_nlist = peer_nb(self.rcut, self.sel, batch)
        self.assertTrue(np.allclose(base_rij, peer_rij))
        self.assertTrue(np.allclose(base_nlist, peer_nlist))

    def test_force_and_virial(self):
        batch = self.dataset.get_batch()
        batch['net_deriv'] = np.random.random(size=[self.batch_size, 192*self.nnei*3])
        rij, nlist, b_force, b_virial, b_atom_virial = forward_nb(self.rcut, self.sel, batch)
        a_force = calc_force(batch['net_deriv'], nlist, self.nloc, self.nall, self.nnei)
        a_virial, a_atom_virial = calc_virial(batch['net_deriv'], rij, nlist, self.nloc, self.nall, self.nnei)
        self.assertTrue(np.allclose(a_force, b_force))
        self.assertTrue(np.allclose(a_virial, b_virial))
        self.assertTrue(np.allclose(a_atom_virial, b_atom_virial))

    def test_force_and_virial_grad(self):
        batch = self.dataset.get_batch()
        batch['force_grad'] = np.random.random(size=[self.batch_size, 192*3])
        batch['virial_grad'] = np.random.random(size=[self.batch_size, 9])
        rij, nlist, b_gf, b_gv = backward_nb(self.rcut, self.sel, batch)
        a_gf = calc_grad_from_force(batch['force_grad'], nlist, self.nloc, self.nnei)
        b_gv = calc_grad_from_virial(batch['virial_grad'], rij, nlist, self.nloc, self.nnei)
        self.assertTrue(np.allclose(a_gf, b_gf))
        self.assertTrue(np.allclose(b_gv, b_gv))


if __name__ == '__main__':
    unittest.main()
