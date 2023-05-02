############################################
#
#  Main
#
###########################################
from rdkit import Chem
import numpy as np
import math
import pandas as pd

from io_pca import write_file_from_list 
#
#  You can change kbond
#
kbond = 5
#
sdf_file='./example.sdf'
prop_list = ['eff_chg','pol','radii','ion1','aff','mass']
datpath = './k'+str(kbond)+'/'
############################
#
#  START PROGRAM
#
############################
nprop = len(prop_list)
sup=Chem.ForwardSDMolSupplier(sdf_file,removeHs=False,sanitize=False)
#
#
#
data = {}
data['file name'] = []
data['atom type'] = []
data['pKa']       = []
for mol in sup:
   mol.UpdatePropertyCache(strict=False)

   mol_name = mol.GetProp('_Name')
   mol_name = mol_name.strip('_')
 
   nat = mol.GetNumAtoms()
   prop_list_tmp = list(mol.GetPropNames())

   cal_prop = {}
   for prop in prop_list:
      cal_prop[prop] = [float(x) for x in mol.GetProp('atom.dprop.'+prop).split()]

   if 'atom.prop.atom_type' in prop_list_tmp:
      atom_type = [x for x in mol.GetProp('atom.prop.atom_type').split()]
   else:
      atom_type = [' ' for i in range(nat)]
#
#  read bond index
#
   tmp_str = mol.GetProp('wiberg').split()
   tmp_float = np.array([float(d) for d in tmp_str])
   bindx = tmp_float.reshape(nat,nat)
##########################################
#
#  WN 
#
#########################################

   d = np.zeros((nat,nat))
   omg = np.zeros((nat,nat))
   for i in range(nat):
      tmp = 1.0/math.sqrt(np.sum(bindx[i,:]))
      d[i,i] = tmp

   omg = d @ bindx @ d

   wn = np.zeros((nat,nat,kbond+1))
   wn[range(nat),range(nat),0] = 1.0
      
   w_max = np.zeros((nat,nat))
   w_max[range(nat),range(nat)] = 1.0

   for ibond in range(kbond):
      tmp = np.zeros((nat,nat))

      tmp[:,:] = omg @ wn[:,:,ibond]
      tmp[range(nat),range(nat)] = 0.0


      for i in range(nat):
         for j in range(nat):
            if tmp[i,j] < w_max[i,j]:
               tmp[i,j] = 0.0

      for i in range(nat):
         for j in range(i):
            val = math.sqrt(tmp[j,i]*tmp[i,j])
            wn[i,j,ibond+1] = val
            wn[j,i,ibond+1] = val

            if val > 0.0:
               w_max[i,j] = val
               w_max[j,i] = val
#
#  prepare data1d
#
   for iat in range(nat):
      atom = mol.GetAtomWithIdx(iat)
      anam = atom.GetSymbol()

      ndim = (kbond+1)*nprop
      if 'csed_charge' in prop_list:
        ndim = ndim + kbond+1
      if 'iso_shield' in prop_list:
        ndim = ndim + kbond+1
      if 'zgrad' in prop_list:
        ndim = ndim + kbond+1

      data1d = np.zeros(ndim)

      jbond = -1
      for prop in prop_list:
         tmp = cal_prop[prop]

         for ibond in range(kbond+1):
            jbond = jbond + 1
      
            val = 0.0
            for jat in range(nat):
               val = val + tmp[jat]*wn[iat,jat,ibond]
#
#           positive and negative data are stored
#
            if prop == 'csed_charge' or prop == 'iso_shield' or prop == 'zgrad' :
               if val >= 0.0:
                  data1d[jbond] = val
               else:
                  data1d[jbond+kbond+1] = -val
#
#           only positive data are stored
#
            elif prop == 'fukui_minus' or prop == 'aff':
               if val >= 0.0:
                  data1d[jbond] = val
#
#           data must be positive
#
            else :
               data1d[jbond] = val

         if prop == 'csed_charge' or prop == 'iso_shield' or prop == 'zgrad' :
            jbond = jbond + kbond + 1
#
#     open files & save data
#
      np.savetxt(datpath+mol_name+'_'+anam+'_'+str(iat+1)+'.dat', data1d, fmt='%.6e')
#
#     check pKa data
#
      pka_str = ' '
      if 'PKA_exp' in prop_list_tmp:
         pka = mol.GetProp('PKA_exp')
         pka_label = mol.GetProp('PKA_label')
         if pka_label == anam+str(iat+1):
            pka_str = pka
#
#     atom_type list
#
      data['file name'].append(mol_name+'_'+anam+'_'+str(iat+1)+'.dat')
      data['atom type'].append(atom_type[iat])
      data['pKa'].append(pka_str)

#
#  save csv file
#
df = pd.DataFrame(data,columns=["file name","atom type","pKa"])
df.to_csv(datpath+'atom_list.csv',index=False)
#
#  save prop_list
#
write_file_from_list('prop_list.log', 'w', prop_list)
