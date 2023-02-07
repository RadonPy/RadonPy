#  Copyright (c) 2022. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

# ******************************************************************************
# ff.descriptor module
# ******************************************************************************

import numpy as np
from scipy.stats import skew, kurtosis
import multiprocessing as MP
import concurrent.futures as confu
from rdkit import Chem
from ..core import poly, utils, const


__version__ = '0.2.3'


class FF_descriptor():
    def __init__(self, ff, charge_max=1.0, charge_min=-1.0, polar=False, polar_max=2.0, deuterium=False,
                stats=['mean', 'std', 'skew', 'kurtosis', 'max', 'min'], **kwargs):
        ignore_pt = ['hw', 'n4', 'nx', 'ny', 'nz', 'n+']
        self.ff = ff
        self.polar = polar
        self.deuterium = deuterium
        self.stats = stats

        # Get FF parameter list
        self.ff_mass = [1.008, 12.011, 14.007, 15.999, 18.998, 30.974, 32.067, 35.453, 79.904, 126.904] # H,C,N,O,F,P,S,Cl,Br,I
        self.ff_mass_D = [1.008, 2.014, 12.011, 14.007, 15.999, 18.998, 30.974, 32.067, 35.453, 79.904, 126.904] # H,C,N,O,F,P,S,Cl,Br,I
        self.ff_epsilon = []
        self.ff_sigma = []
        for key, val in ff.param.pt.items():
            if key in ignore_pt: continue
            self.ff_epsilon.append(val.epsilon)
            self.ff_sigma.append(val.sigma)

        self.ff_k_bond = []
        self.ff_r0 = []
        for val in ff.param.bt.values():
            if val.k < 20: continue
            self.ff_k_bond.append(val.k)
            self.ff_r0.append(val.r0)

        self.ff_k_ang = []
        self.ff_theta0 = []
        for val in ff.param.at.values():
            if val.k < 20: continue
            self.ff_k_ang.append(val.k)
            self.ff_theta0.append(val.theta0)

        self.ff_k_dih = []
        self.ff_phi0 = []
        for val in ff.param.dt.values():
            idx = np.argmax(val.k)
            self.ff_k_dih.append(val.k[idx])
            #self.ff_phi0.append(val.d[idx])

        # Set scaling factors
        self.mass_scale    = self.min_max_scale(self.ff_mass)
        self.charge_scale  = self.min_max_scale([charge_max, charge_min])
        self.epsilon_scale = self.min_max_scale(self.ff_epsilon)
        self.sigma_scale   = self.min_max_scale(self.ff_sigma)
        self.k_bond_scale  = self.min_max_scale(self.ff_k_bond)
        self.r0_scale      = self.min_max_scale(self.ff_r0)
        self.polar_scale   = self.min_max_scale([0.0, polar_max])
        self.k_ang_scale   = self.min_max_scale(self.ff_k_ang)
        self.theta0_scale  = self.min_max_scale(self.ff_theta0)
        self.k_dih_scale   = self.min_max_scale(self.ff_k_dih)
        #self.phi0_scale    = self.min_max_scale(self.ff_phi0)

        self.nk = kwargs.get('nk', 20)
        self.s = kwargs.get('s', None)
        self.s_mass = kwargs.get('s_mass', 0.05)
        self.mu = kwargs.get('mu', None)
        self.mu_mass = kwargs.get('mu_mass', None)
        
        # Setting sigma
        if self.s is None:
            if self.polar:
                self.s = np.array([1/self.nk/2]*9)
            else:
                self.s = np.array([1/self.nk/2]*8)
        elif type(self.s) is float:
            if self.polar:
                self.s = np.array([self.s]*9)
            else:
                self.s = np.array([self.s]*8)

        # Setting mu_mass
        if self.mu_mass is None:
            if self.deuterium:
                self.mu_mass = self.mass_scale.scale(self.ff_mass_D)
            else:
                self.mu_mass = self.mass_scale.scale(self.ff_mass)

        # Setting mu
        if self.mu is None:
            if self.polar:
                self.mu = np.zeros((9, self.nk))
            else:
                self.mu = np.zeros((8, self.nk))
            self.mu[:][:] = np.linspace(0.0, 1.0, self.nk*2+1)[1:-1:2]



    def get_param_list(self, mol, ignoreH=False):
        mass = []
        charge = []
        sigma = []
        epsilon = []
        for atom in mol.GetAtoms():
            if ignoreH and atom.GetSymbol() == 'H':
                continue
            mass.append(atom.GetMass())
            charge.append(atom.GetDoubleProp('AtomicCharge'))
            epsilon.append(atom.GetDoubleProp('ff_epsilon'))
            sigma.append(atom.GetDoubleProp('ff_sigma'))

        k_bond = []
        r0 = []
        polar = []
        for b in mol.GetBonds():
            if ignoreH and (b.GetBeginAtom().GetSymbol() == 'H' or b.GetEndAtom().GetSymbol() == 'H'):
                continue
            k_bond.append(b.GetDoubleProp('ff_k'))
            r0.append(b.GetDoubleProp('ff_r0'))
            if self.polar:
                polar.append(abs(b.GetBeginAtom().GetDoubleProp('AtomicCharge')-b.GetEndAtom().GetDoubleProp('AtomicCharge')))

        k_ang = []
        theta0 = []
        for ang in mol.angles:
            if ignoreH and (mol.GetAtomWithIdx(ang.a).GetSymbol() == 'H' or
                mol.GetAtomWithIdx(ang.b).GetSymbol() == 'H' or mol.GetAtomWithIdx(ang.c).GetSymbol() == 'H'):
                continue
            k_ang.append(ang.ff.k)
            theta0.append(ang.ff.theta0)

        k_dih = []
        #phi0 = []
        for dih in mol.dihedrals:
            if ignoreH and (mol.GetAtomWithIdx(dih.a).GetSymbol() == 'H' or mol.GetAtomWithIdx(dih.b).GetSymbol() == 'H' or
                mol.GetAtomWithIdx(dih.c).GetSymbol() == 'H' or mol.GetAtomWithIdx(dih.d).GetSymbol() == 'H'):
                continue
            idx = np.argmax(dih.ff.k)
            k_dih.append(dih.ff.k[idx])
            #phi0.append(dih.ff.d[idx])

        if self.polar:
            return np.array(mass), np.array(charge), np.array(epsilon), np.array(sigma), np.array(k_bond), np.array(r0), np.array(polar), np.array(k_ang), np.array(theta0), np.array(k_dih)
        else:
            return np.array(mass), np.array(charge), np.array(epsilon), np.array(sigma), np.array(k_bond), np.array(r0), np.array(k_ang), np.array(theta0), np.array(k_dih)


    def get_param_mp(self, smiles, mp=None, cyclic=10, ignoreH=False):

        if mp is None:
            mp = utils.cpu_count()

        c = utils.picklable_const()
        args = [[smi, cyclic, self, c] for smi in smiles]

        with confu.ProcessPoolExecutor(max_workers=mp, mp_context=MP.get_context('spawn')) as executor:
            results = executor.map(get_param_mp_wrapper, args)

        # NaN padding for ff parameter dataset
        vp = max(list(map(len, [x[0] for x in results[:]])))
        bp = max(list(map(len, [x[4] for x in results[:]])))
        if self.polar:
            ap = max(list(map(len, [x[7] for x in results[:]])))
            dp = max(list(map(len, [x[9] for x in results[:]])))
        else:
            ap = max(list(map(len, [x[6] for x in results[:]])))
            dp = max(list(map(len, [x[8] for x in results[:]])))
        p_len = max([vp, bp, ap, dp])
        p_list = np.array(list(map(lambda x: x + [np.nan]*(p_len-len(x)), sum(results, []))))
        if self.polar:
            p_list = p_list.reshape(len(results), 10, p_len)
        else:
            p_list = p_list.reshape(len(results), 9, p_len)

        # Remove all NaN mol (fail to assign)
        smiles = np.array(smiles)
        smi_list = smiles[np.where(np.all(np.all(np.isnan(p_list), axis=2), axis=1))]
        p_list = np.delete(p_list, np.where(np.all(np.all(np.isnan(p_list), axis=2), axis=1)), axis=0)

        return p_list, smi_list


    def ff_summary_statistics(self, mols, ratio=None, ignoreH=False):

        if type(mols) is Chem.Mol:
            mols = [mols]

        if ratio is None:
            ratio = np.array([1]*len(mols))

        mass    = np.array([])
        charge  = np.array([])
        epsilon = np.array([])
        sigma   = np.array([])
        k_bond  = np.array([])
        r0      = np.array([])
        polar   = np.array([])
        k_ang   = np.array([])
        theta0  = np.array([])
        k_dih   = np.array([])
        w_atom  = np.array([])
        w_bond  = np.array([])
        w_angle = np.array([])
        w_dih   = np.array([])

        # Get FF parameters of molecules
        for i, mol in enumerate(mols):
            if self.polar:
                m, q, eps, sig, kb, r, pol, ka, th, kd = self.get_param_list(mol, ignoreH=ignoreH)
            else:
                m, q, eps, sig, kb, r, ka, th, kd = self.get_param_list(mol, ignoreH=ignoreH)
            mass    = np.hstack((mass, m))
            charge  = np.hstack((charge, q))
            epsilon = np.hstack((epsilon, eps))
            sigma   = np.hstack((sigma, sig))
            k_bond  = np.hstack((k_bond, kb))
            r0      = np.hstack((r0, r))
            if self.polar: polar = np.hstack((polar, pol))
            k_ang   = np.hstack((k_ang, ka))
            theta0  = np.hstack((theta0, th))
            k_dih   = np.hstack((k_dih, kd))

            w_atom  = np.hstack((w_atom,  np.array([ratio[i]]*len(m))))
            w_bond  = np.hstack((w_bond,  np.array([ratio[i]]*len(kb))))
            w_angle = np.hstack((w_angle, np.array([ratio[i]]*len(ka))))
            w_dih   = np.hstack((w_dih,   np.array([ratio[i]]*len(kd))))

        # Scaling weights
        w_atom  = w_atom / np.sum(w_atom)
        w_bond  = w_bond / np.sum(w_bond)
        w_angle = w_angle / np.sum(w_angle)
        w_dih   = w_dih / np.sum(w_dih)

        # Calculate summary statistics
        desc = []
        if self.polar:
            feature = [mass, charge, epsilon, sigma, k_bond, r0, polar, k_ang, theta0, k_dih]
            weights = [w_atom, w_atom, w_atom, w_atom, w_bond, w_bond, w_bond, w_angle, w_angle, w_dih]
        else:
            feature = [mass, charge, epsilon, sigma, k_bond, r0, k_ang, theta0, k_dih]
            weights = [w_atom, w_atom, w_atom, w_atom, w_bond, w_bond, w_angle, w_angle, w_dih]

        for f, w in zip(feature, weights):
            ave = np.average(f, weights=w)
            var = np.average((f-ave)**2, weights=w)*(len(f)/(len(f)-1))
            for s in self.stats:
                if s == 'mean':
                    desc.append(ave)
                elif s == 'var':
                    desc.append(var)
                elif s == 'std' or s == 'sd':
                    desc.append(np.sqrt(var))
                elif s == 'skew' or s == 'skewness':
                    desc.append(skew(f))
                    #desc.append(np.average(((f-ave)/np.sqrt(var))**3, weights=w))
                elif s == 'kurt' or s == 'kurtosis':
                    desc.append(kurtosis(f))
                    #desc.append(np.average(((f-ave)/np.sqrt(var))**4, weights=w))
                elif s.find('moment_') == 0:
                    k = int(s.split('_')[1])
                    desc.append(np.average(((f-ave)/np.sqrt(var))**k, weights=w))
                elif s == 'max':
                    desc.append(np.max(f))
                elif s == 'min':
                    desc.append(np.min(f))

        return np.array(desc)


    def ffss_mp(self, smiles, ratio=None, mp=None, cyclic=10, ignoreH=False):

        if ratio is None:
            ratio = np.array([None]*len(smiles))
        if mp is None:
            mp = utils.cpu_count()

        c = utils.picklable_const()
        args = [[smi, ratio[i], cyclic, ignoreH, self, c] for i, smi in enumerate(smiles)]

        with confu.ProcessPoolExecutor(max_workers=mp, mp_context=MP.get_context('spawn')) as executor:
            results = executor.map(ffss_mp_wrapper, args)

        return results


    def ffss_desc_names(self):
        desc_names = []
        if self.polar:
            param_list = ['mass', 'charge', 'epsilon', 'sigma', 'k_bond', 'r0', 'polar', 'k_angle', 'theta0', 'k_dih']
        else:
            param_list = ['mass', 'charge', 'epsilon', 'sigma', 'k_bond', 'r0', 'k_angle', 'theta0', 'k_dih']

        for param in param_list:
            for st in self.stats:
                desc_names.append('%s_%s' % (param, st))

        return desc_names


    def ff_kernel_mean(self, mols, ratio=None, nk=None, kernel=None, s=None, s_mass=None, mu=None, mu_mass=None, ignoreH=False):

        if kernel is None:
            kernel = self.Gaussian

        if nk is None:
            nk = self.nk

        # Setting sigma
        if s is None:
            pass
        elif type(s) is float:
            if self.polar:
                self.s = np.array([s]*9)
            else:
                self.s = np.array([s]*8)
        elif type(s) is list:
            self.s = np.array(s)
        s = self.s

        # Setting sigma of mass
        s_mass = self.s_mass

        # Setting mu
        if mu is None:
            pass
        else:
            self.mu = mu
        center = self.mu

        # Setting mu of mass
        if mu_mass is None:
            pass
        else:
            self.mu_mass = mu_mass
        center_mass = self.mu_mass

        if type(mols) is Chem.Mol:
            mols = [mols]

        if ratio is None:
            ratio = np.array([1]*len(mols))

        mass    = np.array([])
        charge  = np.array([])
        epsilon = np.array([])
        sigma   = np.array([])
        k_bond  = np.array([])
        r0      = np.array([])
        polar   = np.array([])
        k_ang   = np.array([])
        theta0  = np.array([])
        k_dih   = np.array([])
        w_atom  = np.array([])
        w_bond  = np.array([])
        w_angle = np.array([])
        w_dih   = np.array([])

        # Get FF parameters of molecules
        for i, mol in enumerate(mols):
            if self.polar:
                m, q, eps, sig, kb, r, pol, ka, th, kd = self.get_param_list(mol, ignoreH=ignoreH)
            else:
                m, q, eps, sig, kb, r, ka, th, kd = self.get_param_list(mol, ignoreH=ignoreH)
            mass    = np.hstack((mass, m))
            charge  = np.hstack((charge, q))
            epsilon = np.hstack((epsilon, eps))
            sigma   = np.hstack((sigma, sig))
            k_bond  = np.hstack((k_bond, kb))
            r0      = np.hstack((r0, r))
            if self.polar: polar = np.hstack((polar, pol))
            k_ang   = np.hstack((k_ang, ka))
            theta0  = np.hstack((theta0, th))
            k_dih   = np.hstack((k_dih, kd))

            w_atom  = np.hstack((w_atom,  np.array([ratio[i]]*len(m))))
            w_bond  = np.hstack((w_bond,  np.array([ratio[i]]*len(kb))))
            w_angle = np.hstack((w_angle, np.array([ratio[i]]*len(ka))))
            w_dih   = np.hstack((w_dih,   np.array([ratio[i]]*len(kd))))

        # Min-max scaling the FF parameters
        mass    = self.mass_scale.scale(mass)
        charge  = self.charge_scale.scale(charge)
        epsilon = self.epsilon_scale.scale(epsilon)
        sigma   = self.sigma_scale.scale(sigma)
        k_bond  = self.k_bond_scale.scale(k_bond)
        r0      = self.r0_scale.scale(r0)
        if self.polar: polar = self.polar_scale.scale(polar)
        k_ang   = self.k_ang_scale.scale(k_ang)
        theta0  = self.theta0_scale.scale(theta0)
        k_dih   = self.k_dih_scale.scale(k_dih)

        # Calculate kernel mean
        if self.polar:
            mass_km    = self.kernel_mean(mass,    kernel, center_mass, weights=w_atom,  sigma=s_mass)
            charge_km  = self.kernel_mean(charge,  kernel, center[0],   weights=w_atom,  sigma=s[0])
            epsilon_km = self.kernel_mean(epsilon, kernel, center[1],   weights=w_atom,  sigma=s[1])
            sigma_km   = self.kernel_mean(sigma,   kernel, center[2],   weights=w_atom,  sigma=s[2])
            k_bond_km  = self.kernel_mean(k_bond,  kernel, center[3],   weights=w_bond,  sigma=s[3])
            r0_km      = self.kernel_mean(r0,      kernel, center[4],   weights=w_bond,  sigma=s[4])
            polar_km   = self.kernel_mean(polar,   kernel, center[5],   weights=w_bond,  sigma=s[5])
            k_ang_km   = self.kernel_mean(k_ang,   kernel, center[6],   weights=w_angle, sigma=s[6])
            theta0_km  = self.kernel_mean(theta0,  kernel, center[7],   weights=w_angle, sigma=s[7])
            k_dih_km   = self.kernel_mean(k_dih,   kernel, center[8],   weights=w_dih,   sigma=s[8])
            desc = np.hstack([mass_km, charge_km, epsilon_km, sigma_km, k_bond_km, r0_km, polar_km, k_ang_km, theta0_km, k_dih_km])
        else:
            mass_km    = self.kernel_mean(mass,    kernel, center_mass, weights=w_atom,  sigma=s_mass)
            charge_km  = self.kernel_mean(charge,  kernel, center[0],   weights=w_atom,  sigma=s[0])
            epsilon_km = self.kernel_mean(epsilon, kernel, center[1],   weights=w_atom,  sigma=s[1])
            sigma_km   = self.kernel_mean(sigma,   kernel, center[2],   weights=w_atom,  sigma=s[2])
            k_bond_km  = self.kernel_mean(k_bond,  kernel, center[3],   weights=w_bond,  sigma=s[3])
            r0_km      = self.kernel_mean(r0,      kernel, center[4],   weights=w_bond,  sigma=s[4])
            k_ang_km   = self.kernel_mean(k_ang,   kernel, center[5],   weights=w_angle, sigma=s[5])
            theta0_km  = self.kernel_mean(theta0,  kernel, center[6],   weights=w_angle, sigma=s[6])
            k_dih_km   = self.kernel_mean(k_dih,   kernel, center[7],   weights=w_dih,   sigma=s[7])
            desc = np.hstack([mass_km, charge_km, epsilon_km, sigma_km, k_bond_km, r0_km, k_ang_km, theta0_km, k_dih_km])

        return desc


    def ffkm_mp(self, smiles, ratio=None, mp=None, nk=None, kernel=None,
                s=None, s_mass=None, cyclic=10, mu=None, mu_mass=None, ignoreH=False):

        if ratio is None:
            ratio = np.array([None]*len(smiles))
        if mp is None:
            mp = utils.cpu_count()

        c = utils.picklable_const()
        args = [[smi, ratio[i], nk, kernel, s, s_mass, mu, mu_mass, cyclic, ignoreH, self, c] for i, smi in enumerate(smiles)]
    
        with confu.ProcessPoolExecutor(max_workers=mp, mp_context=MP.get_context('spawn')) as executor:
            results = executor.map(ffkm_mp_wrapper, args)

        return results


    def ffkm_desc_names(self, nk=None):
        if nk is None:
            nk = self.nk

        if self.deuterium:
            desc_names = ['mass_H', 'mass_D', 'mass_C', 'mass_N', 'mass_O', 'mass_F', 'mass_P', 'mass_S', 'mass_Cl', 'mass_Br', 'mass_I']
        else:
            desc_names = ['mass_H', 'mass_C', 'mass_N', 'mass_O', 'mass_F', 'mass_P', 'mass_S', 'mass_Cl', 'mass_Br', 'mass_I']

        if self.polar:
            param_list = ['charge', 'epsilon', 'sigma', 'k_bond', 'r0', 'polar', 'k_angle', 'theta0', 'k_dih']
        else:
            param_list = ['charge', 'epsilon', 'sigma', 'k_bond', 'r0', 'k_angle', 'theta0', 'k_dih']

        for param in param_list:
            for i in range(nk):
                desc_names.append('%s_%i' % (param, i))

        return desc_names


    class min_max_scale():
        def __init__(self, l, vmax=1.0, vmin=0.0):
            l = np.array(l)
            l_max = np.max(l)
            l_min = np.min(l)
            self.s = (l_max - l_min) * (vmax - vmin)
            self.l_min = l_min
            self.l_max = l_max
            self.vmin = vmin
            self.vmax = vmax

        def scale(self, l):
            return (np.array(l) - self.l_min) / self.s + self.vmin

        def unscale(self, l):
            return (np.array(l) - self.vmin) * self.s + self.l_min


    def kernel_mean(self, vec, func, centers, weights=None, **kwargs):
        x = vec.reshape([1, -1])
        xm = centers.reshape([-1, 1])
        y = func(x, xm, **kwargs)
        if weights is None:
            km = np.mean(y, axis=1)
        else:
            weights = np.array(weights) / np.sum(weights)
            km = np.sum(y*weights, axis=1)
        return km


    def Gaussian(self, x, xm, **kwargs):
        sigma = kwargs.get('sigma', 0.1)
        return np.exp(-(x - xm)**2/(2*sigma**2))

    def Step(self, x, xm, **kwargs):
        sigma = kwargs.get('sigma', 0.1)
        return np.where(-sigma < (x - xm) <= sigma, 1, 0)

    def Cauchy(self, x, xm, **kwargs):
        sigma = kwargs.get('sigma', 0.1)
        return sigma/(np.pi*((x-xm)**2 + sigma**2))

    def Laplace(self, x, xm, **kwargs):
        sigma = kwargs.get('sigma', 0.1)
        return np.exp(-np.abs(x - xm)/sigma) / (2*sigma)


def ffss_mp_wrapper(args):
    flag = True
    mols = []

    smi, ratio, cyclic, ignoreH, descobj, c = args
    utils.restore_const(c)

    if type(smi) is not list:
        smi = [smi]

    try:
        for smiles in smi:
            if cyclic:
                mol = poly.make_cyclicpolymer(smiles, n=cyclic, return_mol=True)
            else:
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.AddHs(mol)
            result = descobj.ff.ff_assign(mol, charge='gasteiger')
            mols.append(mol)
            if not result:
                flag = False

        if flag:
            desc = descobj.ff_summary_statistics(mols, ratio=ratio, ignoreH=ignoreH)
        else:
            utils.radon_print('Fail to assign force field. %s' % (','.join(smi)), level=2)
            desc = np.full(len(descobj.ffss_desc_names()), np.nan)
    except:
        utils.radon_print('Fail to calculate descriptor. %s' % (','.join(smi)), level=2)
        desc = np.full(len(descobj.ffss_desc_names()), np.nan)

    return desc


def ffkm_mp_wrapper(args):
    flag = True
    mols = []

    smi, ratio, nk, kernel, s, s_mass, mu, mu_mass, cyclic, ignoreH, descobj, c = args
    utils.restore_const(c)

    if type(smi) is not list:
        smi = [smi]

    try:
        for smiles in smi:
            if cyclic:
                mol = poly.make_cyclicpolymer(smiles, n=cyclic, return_mol=True)
            else:
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.AddHs(mol)
            result = descobj.ff.ff_assign(mol, charge='gasteiger')
            mols.append(mol)
            if not result:
                flag = False

        if flag:
            desc = descobj.ff_kernel_mean(mols, ratio=ratio, nk=nk, kernel=kernel,
                                          s=s, s_mass=s_mass, mu=mu, mu_mass=mu_mass, ignoreH=ignoreH)
        else:
            utils.radon_print('Fail to assign force field. %s' % (','.join(smi)), level=2)
            desc = np.full(len(descobj.ffkm_desc_names(nk=nk)), np.nan)

    except:
        utils.radon_print('Fail to calculate descriptor. %s' % (','.join(smi)), level=2)
        desc = np.full(len(descobj.ffkm_desc_names(nk=nk)), np.nan)

    return desc


def get_param_mp_wrapper(args):
    smi, cyclic, descobj, c = args
    utils.restore_const(c)

    try:
        if cyclic:
            mol = poly.make_cyclicpolymer(smi, n=cyclic, return_mol=True)
        else:
            mol = Chem.MolFromSmiles(smi)
            mol = Chem.AddHs(mol)
        result = descobj.ff.ff_assign(mol, charge='gasteiger')
    except:
        result = False

    mass = charge = epsilon = sigma = k_bond = r0 = polar = k_ang = theta0 = k_dih = np.array([np.nan])
    if result:
        if descobj.polar:
            mass, charge, epsilon, sigma, k_bond, r0, polar, k_ang, theta0, k_dih = descobj.get_param_list(mol)
        else:
            mass, charge, epsilon, sigma, k_bond, r0, k_ang, theta0, k_dih = descobj.get_param_list(mol)
        
        mass    = descobj.mass_scale.scale(mass)
        charge  = descobj.charge_scale.scale(charge)
        epsilon = descobj.epsilon_scale.scale(epsilon)
        sigma   = descobj.sigma_scale.scale(sigma)
        k_bond  = descobj.k_bond_scale.scale(k_bond)
        r0      = descobj.r0_scale.scale(r0)
        if descobj.polar:
            polar = descobj.polar_scale.scale(polar)
        k_ang   = descobj.k_ang_scale.scale(k_ang)
        theta0  = descobj.theta0_scale.scale(theta0)
        k_dih   = descobj.k_dih_scale.scale(k_dih)

        if descobj.polar:
            param = [mass.tolist(), charge.tolist(), epsilon.tolist(), sigma.tolist(),
                     k_bond.tolist(), r0.tolist(), polar.tolist(), k_ang.tolist(), theta0.tolist(), k_dih.tolist()]
        else:
            param = [mass.tolist(), charge.tolist(), epsilon.tolist(), sigma.tolist(),
                     k_bond.tolist(), r0.tolist(), k_ang.tolist(), theta0.tolist(), k_dih.tolist()]

    else:
        utils.radon_print('Fail to assign force field. %s' % (smi), level=2)
        if descobj.polar:
            param = np.full((10, 1), np.nan).tolist()
        else:
            param = np.full((9, 1), np.nan).tolist()

    return param

