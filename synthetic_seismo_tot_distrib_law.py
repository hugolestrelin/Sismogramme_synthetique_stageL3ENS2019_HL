#!/user/bin/python
import sys,os
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from numba import njit
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from scipy.stats import norm
import matplotlib.gridspec as gridspec
np.set_printoptions(threshold=sys.maxsize)

# HUGO LESTRELIN - Stage de recherche juin / juillet 2019
# Tuteur = H. S. BHAT
# Co-tutrice = L. BRUHAT

################################################################################
# Paramètres de modélisation :
################################################################################

rayon = 20 # le rayon d'où s'observent les ondes, en kilomètre
ou=20 # Où se localise t-on au niveau du recépteur (à r) afin d'observer l'arrivée des ondes; en degrés.
orientation_voulue = 213 # orientation en degré
Nombre_de_failles = 50 # nombre de failles autour de la faille principale
v=3 # Vitesse moyenne de la propagation de la faille principale vers les failles secondaires en km/s
d=2 # distance entre chaque failles en km
start_time=10 # quand souhaite t-on démarrer le séisme, en secondes
duration=8 # combien de temps souhaite t-on que dure la source time function
aléatoire=0
gauss=1 # si gauss = 1 alors la STF correspond à une Gaussienne, si Gauss = 0 alors c'est un rectangle
# ATTENTION PAS ENCORE UPDATÉ POUR FONCTION RECTANGLE : I.E ONLY GAUSSIAN AVAILABLE
step=50 # Pas virtuel attribué à toutes les opérations sur listes et matrices. De préférence, à ne pas changer, sauf si besoin d'une plus grande résolution.

################################################################################
# Paramètres physiques :
################################################################################

pi=np.pi
rho=3300 # densité moyenne de la croûte
nu=0.3 # ratio de Poisson
y=100*(10**9) # Module de Young
mu=y/(2*(1+nu)) # Coefficient de Lamé
lambd=(y*nu)/((1+nu)*1-2*nu) # Coefficient de Lamé
alpha=np.sqrt((lambd+2*mu)/rho) # Vitesse théorique des ondes P
beta=np.sqrt(mu/rho) # Vitesse théorique des ondes S
r=rayon*1000

################################################################################
# Directions et composantes de propagation des ondes
################################################################################

orient_step=orientation_voulue*((step-1)/360) # normalisation de l'orientation en degrés sur le pas de modélisation

def orientation_représentation(x): # fonction de représentation de la faille sur le cercle trigonométrique
    return np.tan(orientation_voulue*(2*pi/360))*x + 0

te_ref=np.zeros(step) #theta de référence, i.e avec un faille orientée N-S
te=np.zeros(step)
for i in range(step):
    te_ref[i]=(step-1-i)*2*pi/(step-1)

if (orientation_voulue<270): # theta d'orientation voulue pour faille principale
    for i in range(step):
        te[i]=te_ref[i-int(round(90*((step-1)/360)))-int(round(orient_step))]
elif (orientation_voulue>270):
    for i in range(step):
        te[i-int(round((step-1)-orient_step))]=te_ref[i]

ph=np.zeros(step) #phi, n'intervient de toute façon pas beaucoup car nous nous plaçons seulement selon x1 et x3
for i in range(step):
    ph[i]=i*2*pi/(step-1)

rd=np.zeros((3,step,step)) # ^r
for i in range(step):
    for n in range(step):
        rd[0,i,n] = np.sin(te[i])*np.cos(ph[n])
        rd[1,i,n] = np.sin(te[i])*np.sin(ph[n])
        rd[2,i,n] = np.cos(te[i])

ted=np.zeros((3,step,step)) # ^theta
for i in range(step):
    for n in range(step):
        ted[0,i,n] = np.cos(te[i])*np.cos(ph[n])
        ted[1,i,n] = np.cos(te[i])*np.sin(ph[n])
        ted[2,i,n] = -np.sin(te[i])

phid=np.zeros((3,step,step)) # ^phi
for i in range(step):
    for n in range(step):
        phid[0,i,n] = -np.sin(ph[n])
        phid[1,i,n] = np.cos(ph[n])
        phid[2,i,n] = 0

################################################################################
############## GÉNÉRALISATION aux failles secondaires ##########################
################################################################################
# Ajout failles secondaires

################################################################################
# LOI DE DISTRIBUTION DES FAILLES
################################################################################

coefficient_directeur=-1
def durées_f(y_val):
    return ((y_val-(Nombre_de_failles+1))/coefficient_directeur)*(duration/Nombre_de_failles)

#plt.plot([durées_f(y_val) for y_val in range(Nombre_de_failles)],'k',lw=2)

coefficient_directeur=-0.5
def durées_f(y_val):
    return ((y_val-(50+1))/coefficient_directeur)#*(8/50)*-coefficient_directeur
plt.plot([durées_f(y_val) for y_val in range(50)],'k',lw=2)

################################################################################
################################################################################

orientation_failles=np.zeros(Nombre_de_failles) # vecteur des orientations
orient_f_step=np.zeros(Nombre_de_failles)
tef=np.zeros((Nombre_de_failles,step)) # matrice des orientations de theta
phf=np.zeros((Nombre_de_failles,step)) # matrice des orientations de theta

for n in range(Nombre_de_failles):
    orientation_failles[n]=np.random.randint(0,360)
    orient_f_step[n]=orientation_failles[n]*((step-1)/360)
    if (orientation_failles[n]<270):
        for i in range(step):
            tef[n,i]=te_ref[i-int(round(90*((step-1)/360)))-int(round(orient_f_step[n]))]
            phf[n,i]=i*2*pi/(step-1)
    elif (orientation_failles[n]>270):
        for i in range(step):
            tef[n,i-int(round((step-1)-orient_f_step[n]))]=te_ref[i]
            phf[n,i]=i*2*pi/(step-1)


rdf=np.zeros((Nombre_de_failles,3,step,step)) # ^r
tedf=np.zeros((Nombre_de_failles,3,step,step)) # ^theta
phidf=np.zeros((Nombre_de_failles,3,step,step)) # ^phi
for w in range(Nombre_de_failles):
    for i in range(step):
        for n in range(step):
            rdf[w,0,i,n] = np.sin(tef[w,i])*np.cos(phf[w,n])
            rdf[w,1,i,n] = np.sin(tef[w,i])*np.sin(phf[w,n])
            rdf[w,2,i,n] = np.cos(tef[w,i])
            tedf[w,0,i,n] = np.cos(tef[w,i])*np.cos(phf[w,n])
            tedf[w,1,i,n] = np.cos(tef[w,i])*np.sin(phf[w,n])
            tedf[w,2,i,n] = -np.sin(tef[w,i])
            phidf[w,0,i,n] = -np.sin(phf[w,n])
            phidf[w,1,i,n] = np.cos(phf[w,n])
            phidf[w,2,i,n] = 0

################################################################################
############## GÉNÉRALISATION aux failles secondaires ##########################
################################################################################

# Pas de temps
################################################################################

tps=[]
tps=np.zeros(step*10)
for i in range(len(tps)-1):
    tps[i+1]=tps[i]+0.0004*len(tps)

################################################################################
# Paramètres pour la généralisation de la STF en fonction du nombre de faille
################################################################################

delay=d/v
#    starting_time[i+1]=starting_time[i]+delay

################################################################################
############## GÉNÉRALISATION aux failles secondaires ##########################
################################################################################
# Paramètres pour la généralisation de la STF en fonction du nombre de failles entourant la faille principale

rf=np.zeros(Nombre_de_failles)# différentes distances pour les différentes failles secondaires activées

positions_failles=np.zeros(Nombre_de_failles) # position aléatoire le long de la faille principale
temps_de_départ_1_3=np.zeros(int(round(Nombre_de_failles/3))) # temps (donc position) à laquelle faille (STF) se déclanche
temps_de_départ_1_3[0]=start_time
temps_de_départ_2_3=np.zeros(int(round(Nombre_de_failles*(2/3)))) # temps (donc position) à laquelle faille (STF) se déclanche
temps_de_départ_2_3[0]=start_time
temps_de_départ_3_3=np.zeros(Nombre_de_failles) # temps (donc position) à laquelle faille (STF) se déclanche
temps_de_départ_3_3[0]=start_time
durées=np.zeros(Nombre_de_failles) # durée aléatoire du STF de la faille secondaire concernée

    #for i in range(Nombre_de_failles-1):
#temps_de_départ_3_3[i+1]=temps_de_départ_3_3[i]+durées_f(Nombre_de_failles)
#for i in range(int(round(Nombre_de_failles*(2/3)))-1):
#   temps_de_départ_2_3[i+1]=temps_de_départ_2_3[i]+durées_f(Nombre_de_failles*(2/3))
#for i in range(int(round(Nombre_de_failles/3))-1):
# temps_de_départ_1_3[i+1]=temps_de_départ_1_3[i]+durées_f(Nombre_de_failles/3)
#  rf[i]=np.random.uniform(r-0.01*r,r+0.01*r)


for i in range(Nombre_de_failles-1):
    temps_de_départ_3_3[i+1]=np.random.uniform(start_time,start_time+duration)
for i in range(int(round(Nombre_de_failles*(2/3)))-1):
    temps_de_départ_2_3[i+1]=np.random.uniform(start_time,start_time+duration)
for i in range(int(round(Nombre_de_failles/3))-1):
    temps_de_départ_1_3[i+1]=np.random.uniform(start_time,start_time+duration)
    rf[i]=np.random.uniform(r-0.01*r,r+0.01*r)

################################################################################
############## GÉNÉRALISATION aux failles secondaires ##########################
################################################################################

################################################################################
# Différentes fonctions pour la STF :
################################################################################


M0=np.zeros(len(tps))
M0p=np.zeros(len(tps))
M0d=np.zeros(len(tps))
M0g=np.zeros(len(tps))

M0_a=np.zeros(len(tps))
M0p_a=np.zeros(len(tps))
M0d_a=np.zeros(len(tps))
M0g_a=np.zeros(len(tps))

M0dif=np.zeros(len(tps))

M0gd=np.zeros(len(tps))

M0a=np.zeros(len(tps)) # STF avec retard lié à la vitesse alpha
M0b=np.zeros(len(tps)) # STF avec retard lié à la vitesse beta
M0ad=np.zeros(len(tps))
M0bd=np.zeros(len(tps))


@njit()
def Gaussian(M0): # Fonction Gaussienne
    for i in range(len(tps)):
        if (start_time<=tps[i]<=start_time+duration):
            mu=(start_time+start_time+duration)/2 # moyenne
            e=((duration)/5.5) # écart-type + facteur d'atténuation : n*XXX
            M0[i]=(1 / (((e*np.sqrt(2*pi)))) * np.exp((-1/2)*(((tps[i]-mu)/e)**2)))*(10**19)
    return M0
################################################################################
#DÉRIVÉE => pour vélocimètre

#@njit()
#def Gaussiand(M0dif): # Fonction Gaussienne dérivée
#    for n in range(Fnbr-1):
#        for i in range(len(tps)):
#            if (starting_time[n]<=tps[i]<=starting_time[n]+duration):
#                mu=(starting_time[n]+starting_time[n]+duration)/2 # moyenne
#               e=((duration)/5.5)+n*0.25 # écart-type + facteur d'atténuation : n*XXX
#                M0dif[n,i]= ((tps[i]-mu)/(e**2)) * (1 / (((e*np.sqrt(2*pi)))) * np.exp((-1/2)*(((tps[i]-mu)/e)**2)))*(10**19)
#    return M0dif

#M0gd=Gaussiand(M0dif)

mud=(start_time+start_time+duration)/2 # moyenne
ed=((duration)/5.5) # écart-type + facteur d'atténuation : n*XXX
for i in range(len(tps)):
    if (start_time<=tps[i]<=start_time+duration):
        M0dif[i]= ((tps[i]-mud)/(ed**2)) * (1 / (((ed*np.sqrt(2*pi)))) * np.exp((-1/2)*(((tps[i]-mud)/ed)**2)))*(10**19)

#@njit()
#def sum_gaussd(M0gd): # Somme des Gaussiennes
#    for i in range(len(tps)):
#        for n in range(1,Fnbr):
#            if (M0gd[n,i] > M0gd[0,i]):
#                M0gd[0,i]=M0gd[0,1]+M0gd[n,i]
#    return M0gd

################################################################################
#test en différences finies : => ne fonctionne pas car step temps trop faible, formation d'un "pic"
#M0gdfd=np.zeros(len(tps))
#M0d=sum_gauss(M0g) ##ATTENTION !!!!!!!!

#for i in range(len(tps)-1):
#    M0gdfd[i]=(M0d[0,i+1]-M0d[0,i])/(tps[i+1]-tps[i])


#DÉRIVÉE
################################################################################

M0g=Gaussian(M0)

@njit()
def sum_gauss(M0g): # Somme des Gaussiennes
    for i in range(len(tps)):
        if (M0g[i] > M0g[i]):
            M0g[i]=M0g[1]+M0g[i]
    return M0g




if (aléatoire==0):
    if (gauss==1):
        for i in range(len(tps)):
            if (i >= r/alpha):
                M0a[i]=M0g[i-int(round(r/alpha))]
                M0ad[i]=M0dif[i-int(round(r/alpha))]
        for i in range(len(tps)):
            if (i >= r/beta):
                M0b[i]=M0g[i-int(round(r/beta))]
                M0bd[i]=M0dif[i-int(round(r/beta))]


################################################################################
############## GÉNÉRALISATION aux failles secondaires ##########################
################################################################################
# Généralisation de la STF pour les failles aux alentours :

M0f_1_3=np.zeros((int(round(Nombre_de_failles/3)),len(tps))) # tenseur de moment
M0fd_1_3=np.zeros((int(round(Nombre_de_failles/3)),len(tps)))
M0pf_1_3=np.zeros((int(round(Nombre_de_failles/3)),len(tps)))
M0df_1_3=np.zeros((int(round(Nombre_de_failles/3)),len(tps)))
M0dfd_1_3=np.zeros((int(round(Nombre_de_failles/3)),len(tps)))
M0gf_1_3=np.zeros((int(round(Nombre_de_failles/3)),len(tps)))
M0gfd_1_3=np.zeros((int(round(Nombre_de_failles/3)),len(tps)))

M0af_1_3=np.zeros((int(round(Nombre_de_failles/3)),len(tps)))
M0bf_1_3=np.zeros((int(round(Nombre_de_failles/3)),len(tps)))
M0afd_1_3=np.zeros((int(round(Nombre_de_failles/3)),len(tps)))
M0bfd_1_3=np.zeros((int(round(Nombre_de_failles/3)),len(tps)))

M0f_2_3=np.zeros((int(round(Nombre_de_failles*(2/3))),len(tps)))
M0fd_2_3=np.zeros((int(round(Nombre_de_failles*(2/3))),len(tps)))
M0pf_2_3=np.zeros((int(round(Nombre_de_failles*(2/3))),len(tps)))
M0df_2_3=np.zeros((int(round(Nombre_de_failles*(2/3))),len(tps)))
M0dfd_2_3=np.zeros((int(round(Nombre_de_failles*(2/3))),len(tps)))
M0gf_2_3=np.zeros((int(round(Nombre_de_failles*(2/3))),len(tps)))
M0gfd_2_3=np.zeros((int(round(Nombre_de_failles*(2/3))),len(tps)))

M0af_2_3=np.zeros((int(round(Nombre_de_failles*(2/3))),len(tps)))
M0bf_2_3=np.zeros((int(round(Nombre_de_failles*(2/3))),len(tps)))
M0afd_2_3=np.zeros((int(round(Nombre_de_failles*(2/3))),len(tps)))
M0bfd_2_3=np.zeros((int(round(Nombre_de_failles*(2/3))),len(tps)))

M0f_3_3=np.zeros((Nombre_de_failles,len(tps)))
M0fd_3_3=np.zeros((Nombre_de_failles,len(tps)))
M0pf_3_3=np.zeros((Nombre_de_failles,len(tps)))
M0df_3_3=np.zeros((Nombre_de_failles,len(tps)))
M0dfd_3_3=np.zeros((Nombre_de_failles,len(tps)))
M0gf_3_3=np.zeros((Nombre_de_failles,len(tps)))
M0gfd_3_3=np.zeros((Nombre_de_failles,len(tps)))

M0af_3_3=np.zeros((Nombre_de_failles,len(tps)))
M0bf_3_3=np.zeros((Nombre_de_failles,len(tps)))
M0afd_3_3=np.zeros((Nombre_de_failles,len(tps)))
M0bfd_3_3=np.zeros((Nombre_de_failles,len(tps)))
# Ma et Mb pour sum de STF
M0af_tot=np.zeros(len(tps))
M0bf_tot=np.zeros(len(tps))


# Ma et Mb pour sum de dispalcement
M0af=np.zeros((Nombre_de_failles,len(tps)))
M0bf=np.zeros((Nombre_de_failles,len(tps)))
M0afd=np.zeros((Nombre_de_failles,len(tps)))
M0bfd=np.zeros((Nombre_de_failles,len(tps)))

def triangle_difff(M0pf): # Fonction carré
    for n in range(len(temps_de_départ)):
        #if (n<((1/3)*F_nbr)):
        for i in range(len(tps)):
            if (temps_de_départ[n]<=tps[i]<=(temps_de_départ[n]+durées[n])):
                M0pf[n,i]=(5*10**19)#-((n)*3*10**17)#-(n*13.5**16) # second terme = atténuation ### WATCH OUT ICI QUE DÉTERMINE AMPLITUDE, DOIT DIMINUER POUR FAILLES SECONDAIRES
    #else:
    #for i in range(len(tps)):
#  if (starting_time[n]<=tps[i]<=(starting_time[n]+duration)):
#     M0p[0,i]=(10**19)-(n*16*10**16)
    return M0pf

@njit()
def Gaussianf(M0f): # Fonction Gaussienne
    for n in range(Nombre_de_failles):
        atténuation_aléatoire=np.random.uniform(0.1*10**18,1.58*10**18) # ICI QUE FACTEUR ATTÉNUATION BIEN PLACÉ !!!!!!!!!!!!!!!!!!!!!!!!!
        for i in range(len(tps)):
            if (temps_de_départ[n]<=tps[i]<=(temps_de_départ[n]+durées[n])):
                mu=(temps_de_départ[n]+temps_de_départ[n]+durées[n])/2 # moyenne
                e=((durées[n])/(2.2*durées[n]))#+n*0.25 # écart-type + facteur d'atténuation : n*XXX #### ATTENTION PARAMÈTRE DANGEREUX !!!! #### normalement c'est bon car proportionnalité liée à vecteur durée...
                M0f[n,i]=(1 / (((e*np.sqrt(2*pi)))) * np.exp((-1/2)*(((tps[i]-mu)/e)**2)))*((2*10**18)-atténuation_aléatoire)
    return M0f

#@njit()
def Gaussianf_3_3(moment_tensor3,nombre_failles3,temps_de_départ3): # Fonction Gaussienne
    for n in range(nombre_failles3):
        #atténuation_aléatoire=np.random.uniform(0.1*10**18,1.58*10**18) # ICI QUE FACTEUR ATTÉNUATION BIEN PLACÉ !!!!!!!!!!!!!!!!!!!!!!!!!
        for i in range(len(tps)):
            if (temps_de_départ3[n]<=tps[i]<=(temps_de_départ3[n]+durées_f(nombre_failles3))):
                mu=(temps_de_départ3[n]+temps_de_départ3[n]+durées_f(nombre_failles3))/2 # moyenne
                e=((durées_f(nombre_failles3))/(2.2*durées_f(nombre_failles3)))#+n*0.25 # écart-type + facteur d'atténuation : n*XXX #### ATTENTION PARAMÈTRE DANGEREUX !!!! #### normalement c'est bon car proportionnalité liée à vecteur durée...
                moment_tensor3[n,i]=(1 / (((e*np.sqrt(2*pi)))) * np.exp((-1/2)*(((tps[i]-mu)/e)**2)))*((2*10**18))#-atténuation_aléatoire)
    return moment_tensor3

M0f_1_3=Gaussianf_3_3(M0f_1_3,int(round(Nombre_de_failles/3)),temps_de_départ_1_3)
M0f_2_3=Gaussianf_3_3(M0f_2_3,int(round(Nombre_de_failles*(2/3))),temps_de_départ_2_3)
M0f_3_3=Gaussianf_3_3(M0f_3_3,int(round(Nombre_de_failles)),temps_de_départ_3_3)

################################################################################
#si besoin de faire somme des STF ???

#for i in range(len(tps)):
# MMf[i]=M0d[0,i]
#for i in range(len(tps)):
# for n in range(Nombre_de_failles):
#    MMf[i]=MMf[i]+M0pf[n,i]

#@njit()
#def sum_gauss(M0gf): # Somme des Gaussiennes
#    for i in range(len(tps)):
#        for n in range(1,Fnbr):
#            if (M0g[n,i] > M0g[0,i]):
#                M0g[0,i]=M0g[0,1]+M0g[n,i]
#    return M0gf

################################################################################
#aléatoire

def triangle_diff_af(M0p_af): # Fonction carré
    for n in range(Nombre_de_failles):
        for i in range(len(tps)):
            if (temps_de_départ[n]<=tps[i]<=(temps_de_départ[n]+durées[n])):
                M0p_af[n,i]=(10**19)-(n*np.random.uniform(10**15,9*10**17)) # second terme = atténuation
    return M0p_af

@njit()
def Gaussian_af(M0_af): # Fonction Gaussienne avec valeurs aléatoires
    for n in range(Nombre_de_failles):
        e=((durées[n])/(2.2*durées[n]))+ (np.random.uniform((durées[n])*0.009,(durées[n])*0.09)) # EMPIRIQUE
        mu=(temps_de_départ[n]+temps_de_départ[n]+durées[n])/2
        for i in range(len(tps)):
            if (temps_de_départ[n]<=tps[i]<=(temps_de_départ[n]+durées[n])):
                M0_af[n,i]=(1 / (((e*np.sqrt(2*pi)))) * np.exp((-1/2)*(((tps[i]-mu)/e)**2)))*(2.4*10**18) # EMPIRIQUE
    return M0_af

#M0g_af=Gaussian_af(M0_af)

################################################################################
#si besoin de faire somme des STF ???

#@njit()
#def sum_gauss_a(M0g_a): # Somme des Gaussiennes en aléatoire
#   for i in range(len(tps)):
#        for n in range(1,Fnbr):
#           if (M0g_a[n,i] > M0g_a[0,i]):
#               M0g_a[0,i]=M0g_a[0,1]+M0g_a[n,i]
#   return M0g_a
################################################################################

################################################################################
#DÉRIVÉE => pour vélocimètre

#@njit()
def Gaussianfd_3_3(tenseur_moment3,nombre_failles3,temps_de_départ3): # Fonction Gaussienne
    for n in range(nombre_failles3):
        #atténuation_aléatoire=np.random.uniform(0.1*10**18,1.58*10**18) # ICI QUE FACTEUR ATTÉNUATION BIEN PLACÉ !!!!!!!!!!!!!!!!!!!!!!!!!
        for i in range(len(tps)):
            if (temps_de_départ3[n]<=tps[i]<=(temps_de_départ3[n]+durées_f(nombre_failles3))):
                mu=(temps_de_départ3[n]+temps_de_départ3[n]+durées_f(nombre_failles3))/2 # moyenne
                e=((durées_f(nombre_failles3)/(2.2*durées_f(nombre_failles3))))#+n*0.25 # écart-type + facteur d'atténuation : n*XXX #### ATTENTION PARAMÈTRE DANGEREUX !!!! #### normalement c'est bon car proportionnalité liée à vecteur durée...
                tenseur_moment3[n,i]=((tps[i]-mu)/(e**2)) * (1 / (((e*np.sqrt(2*pi)))) * np.exp((-1/2)*(((tps[i]-mu)/e)**2)))*((2*10**18))#-atténuation_aléatoire)
    return tenseur_moment3
                   
M0fd_1_3=Gaussianfd_3_3(M0fd_1_3,int(round(Nombre_de_failles/3)),temps_de_départ_1_3)
M0fd_2_3=Gaussianfd_3_3(M0fd_2_3,int(round(Nombre_de_failles*(2/3))),temps_de_départ_2_3)
M0fd_3_3=Gaussianfd_3_3(M0fd_3_3,int(round(Nombre_de_failles)),temps_de_départ_3_3)

#M0gfd=Gaussianfd(M0fd) #????

#DÉRIVÉE => pour vélocimètre
################################################################################

def passage_a(tenseur_moment,tenseur_mom_a,nombre_failles3):
    for n in range(nombre_failles3):
        for i in range(len(tps)):
            if (i >= r/alpha):
                tenseur_mom_a[n,i]=tenseur_moment[n,i-int(round(r/alpha))]
    return tenseur_mom_a

M0af_1_3=passage_a(M0f_1_3,M0af_1_3,int(round(Nombre_de_failles/3)))
M0af_2_3=passage_a(M0f_2_3,M0af_2_3,int(round(Nombre_de_failles*(2/3))))
M0af_3_3=passage_a(M0f_3_3,M0af_3_3,int(round(Nombre_de_failles)))

def passage_b(tenseur_moment,tenseur_mom_b,nombre_failles3):
    for n in range(nombre_failles3):
        for i in range(len(tps)):
            if (i >= r/beta):
                tenseur_mom_b[n,i]=tenseur_moment[n,i-int(round(r/beta))]
    return tenseur_mom_b

M0bf_1_3=passage_b(M0f_1_3,M0bf_1_3,int(round(Nombre_de_failles/3)))
M0bf_2_3=passage_b(M0f_2_3,M0bf_2_3,int(round(Nombre_de_failles*(2/3))))
M0bf_3_3=passage_b(M0f_3_3,M0bf_3_3,int(round(Nombre_de_failles)))

def passage_a_d(tenseur_mom_der,tenseur_mom_der_a,nombre_failles3):
    for n in range(nombre_failles3):
        for i in range(len(tps)):
            if (i >= r/alpha):
                tenseur_mom_der_a[n,i]=tenseur_mom_der[n,i-int(round(r/alpha))]
    return tenseur_mom_der_a

M0afd_1_3=passage_a_d(M0fd_1_3,M0afd_1_3,int(round(Nombre_de_failles/3)))
M0afd_2_3=passage_a_d(M0fd_2_3,M0afd_2_3,int(round(Nombre_de_failles*(2/3))))
M0afd_3_3=passage_a_d(M0fd_3_3,M0afd_3_3,int(round(Nombre_de_failles)))

def passage_b_d(tenseur_mom_der,tenseur_mom_der_b,nombre_failles3):
    for n in range(nombre_failles3):
        for i in range(len(tps)):
            if (i >= r/beta):
                tenseur_mom_der_b[n,i]=tenseur_mom_der[n,i-int(round(r/beta))]
    return tenseur_mom_der_b

M0bfd_1_3=passage_b_d(M0fd_1_3,M0bfd_1_3,int(round(Nombre_de_failles/3)))
M0bfd_2_3=passage_b_d(M0fd_2_3,M0bfd_2_3,int(round(Nombre_de_failles*(2/3))))
M0bfd_3_3=passage_b_d(M0fd_3_3,M0bfd_3_3,int(round(Nombre_de_failles)))


################################################################################
############## GÉNÉRALISATION aux failles secondaires ##########################
################################################################################

################################################################################
# Solution pour le déplacement d'ondes élastiques en "far-field" :
################################################################################

u=np.zeros((3,len(tps),len(te),len(ph))) # Matrice de déplacement total
up=np.zeros((3,len(tps),len(te),len(ph))) # Matrice de déplacement pour les ondes P
us=np.zeros((3,len(tps),len(te),len(ph))) # Matrice de déplacement pour les ondes S

@njit()
def sismo_total(u, rd, ted, phid, r, te, phi, M0a, M0b): # Solution pour le déplacement total
    for d in range(3):
        for t in range(len(tps)):
            for i in range(len(te)):
                for n in range(len(ph)):
                    u[d,t,i,n] = (1 / (4*pi*rho*alpha**3)) * (np.sin(2*te[i])*np.cos(phi[n])*rd[d,i,n]) * (1/r) * M0a[t] + (1 / (4*pi*rho*beta**3)) * (np.cos(2*te[i])*np.cos(phi[n])*ted[d,i,n] - np.cos(te[i])*np.sin(phi[n])*phid[d,i,n]) * (1/r) * M0b[t]
    return u

################################################################################
############## GÉNÉRALISATION aux failles secondaires ##########################
################################################################################

uf_1_3=np.zeros((int(round(Nombre_de_failles/3)),3,len(tps),len(te),len(ph))) # pour failles secondaires
uf_2_3=np.zeros((int(round(Nombre_de_failles*(2/3))),3,len(tps),len(te),len(ph)))
uf_3_3=np.zeros((Nombre_de_failles,3,len(tps),len(te),len(ph)))

#@njit()
def sismo_total_f_3_3(nombre_failles3,matrice_u,tenseur_mom_a,tenseur_mom_b): # Solution pour le déplacement total
    for w in range(nombre_failles3):
        for d in range(3):
            for t in range(len(tps)):
                for i in range(len(te)):
                    for n in range(len(ph)):
                        matrice_u[w,d,t,i,n] = (1 / (4*pi*rho*alpha**3)) * (np.sin(2*te[i])*np.cos(ph[n])*rd[d,i,n]) * (1/r) * tenseur_mom_a[t] + (1 / (4*pi*rho*beta**3)) * (np.cos(2*te[i])*np.cos(ph[n])*ted[d,i,n] - np.cos(te[i])*np.sin(ph[n])*phid[d,i,n]) * (1/r) * tenseur_mom_b[t]
    return matrice_u


      
uf_1_3=sismo_total_f_3_3(int(round(Nombre_de_failles/3)), uf_1_3, M0af_1_3, M0bf_1_3)
uf_2_3=sismo_total_f_3_3(int(round(Nombre_de_failles*(2/3))),uf_2_3, M0af_2_3, M0bf_2_3,tps,te,ph,rd,ted,phid)
uf_3_3=sismo_total_f_3_3(int(round(Nombre_de_failles)),uf_3_3, M0af_3_3, M0bf_3_3,tps,te,ph,rd,ted,phid)

################################################################################
#DÉRIVÉE => pour vélocimètre

vd=np.zeros((3,len(tps),len(te),len(ph)))

@njit()
def sismo_totald(vd, rd, ted, phid, r, te, phi, M0ad, M0bd): # Solution pour le déplacement total
    for d in range(3):
        for t in range(len(tps)):
            for i in range(len(te)):
                for n in range(len(ph)):
                    vd[d,t,i,n] = (1 / (4*pi*rho*alpha**3)) * (np.sin(2*te[i])*np.cos(phi[n])*rd[d,i,n]) * (1/r) * M0ad[t] + (1 / (4*pi*rho*beta**3)) * (np.cos(2*te[i])*np.cos(phi[n])*ted[d,i,n] - np.cos(te[i])*np.sin(phi[n])*phid[d,i,n]) * (1/r) * M0bd[t]
    return vd

################################################################################
############## GÉNÉRALISATION aux failles secondaires ##########################
################################################################################

# Décomposition du déplacement :

@njit()
def sismo_P(u, rd, ted, phid, r, te, phi, M0a): # Solution pour le déplacement des ondes P
    for d in range(3):
        for t in range(len(tps)):
            for i in range(len(te)):
                for n in range(len(ph)):
                    u[d,t,i,n] = (1 / (4*pi*rho*alpha**3)) * (np.sin(2*te[i])*np.cos(phi[n])*rd[d,i,n]) * (1/r) * M0a[t]
    return u

@njit()
def sismo_S(u, rd, ted, phid, r, te, phi, M0b): # Solution pour le déplacement des ondes S
    for d in range(3):
        for t in range(len(tps)):
            for i in range(len(te)):
                for n in range(len(ph)):
                    u[d,t,i,n] = (1 / (4*pi*rho*beta**3)) * (np.cos(2*te[i])*np.cos(phi[n])*ted[d,i,n] - np.cos(te[i])*np.sin(phi[n])*phid[d,i,n]) * (1/r) * M0b[t]
    return u

################################################################################
#DÉRIVÉE pour ensemlbe failles

vf_1_3=np.zeros((int(round(Nombre_de_failles/3)),3,len(tps),len(te),len(ph))) # pour failles secondaires
vf_2_3=np.zeros((int(round(Nombre_de_failles*(2/3))),3,len(tps),len(te),len(ph)))
vf_3_3=np.zeros((Nombre_de_failles,3,len(tps),len(te),len(ph)))

#@njit()
def sismo_totald_f_3_3(nombre_failles3,matrice_v,der_tenseur_mom_a,der_tenseur_mom_b): # Solution pour le déplacement total
    for w in range(nombre_failles3):
        for d in range(3):
            for t in range(len(tps)):
                for i in range(len(te)):
                    for n in range(len(ph)):
                        matrice_v[w,d,t,i,n] = (1 / (4*pi*rho*alpha**3)) * (np.sin(2*te[i])*np.cos(ph[n])*rd[d,i,n]) * (1/r) * der_tenseur_mom_a[t] + (1 / (4*pi*rho*beta**3)) * (np.cos(2*te[i])*np.cos(ph[n])*ted[d,i,n] - np.cos(te[i])*np.sin(ph[n])*phid[d,i,n]) * (1/r) * der_tenseur_mom_b[t]
    return matrice_v

vf_1_3=sismo_totald_f_3_3(int(round(Nombre_de_failles/3)),vf_1_3, M0afd_1_3, M0bfd_1_3)
vf_2_3=sismo_totald_f_3_3(int(round(Nombre_de_failles*(2)/3)),vf_2_3, M0afd_2_3, M0bfd_2_3)
vf_3_3=sismo_totald_f_3_3(int(round(Nombre_de_failles)),vf_3_3, M0afd_3_3, M0bfd_3_3)

################################################################################
# Remplissage des matrices :

u=sismo_total(u, rd, ted, phid, r, te, ph, M0a, M0b)
vd=sismo_totald(vd, rd, ted, phid, r, te, ph, M0ad, M0bd)
up=sismo_P(up, rd, ted, phid, r, te, ph, M0a)
us=sismo_S(us, rd, ted, phid, r, te, ph, M0b)

################################################################################
################################################################################
# Réalisation des figures :
################################################################################
################################################################################

where=orientation_voulue*((step-1)/360)
axcolor = 'lightgreen'

gs = gridspec.GridSpec(4, 4)

fig = plt.figure() # pour le plot en 3D
#ax = fig.gca(projection='3d')

x=u[0,:,int(round(where))-1,0]
y=u[1,:,int(round(where))-1,0]
z=u[2,:,int(round(where))-1,0]
xp=up[0,:,int(round(where))-1,0]
zp=up[2,:,int(round(where))-1,0]
xs=us[0,:,int(round(where))-1,0]
zs=us[2,:,int(round(where))-1,0]
xd=u[0,:,:,0]
yd=u[1,:,:,0]
zd=u[2,:,:,0]
t=tps[:]
xv=vd[0,:,int(round(where))-1,0]
yv=vd[1,:,int(round(where))-1,0]
zv=vd[2,:,int(round(where))-1,0]
xvf=vfd[0,:,int(round(where))-1,0]
yvf=vfd[1,:,int(round(where))-1,0]
zvf=vfd[2,:,int(round(where))-1,0]

u_tot=np.zeros((3,len(tps),len(te),len(ph)))
u_tot_1_3=np.zeros((3,len(tps),len(te),len(ph)))
u_tot_2_3=np.zeros((3,len(tps),len(te),len(ph)))
u_tot_3_3=np.zeros((3,len(tps),len(te),len(ph)))

for r in range(3):
    for i in range(len(tps)):
        for n in range(len(te)):
            u_tot[r,i,n,0]=u[r,i,n,0]

for r in range(3):
    for n in range(int(round(Nombre_de_failles/3))):
        for i in range(len(tps)):
            for k in range(len(te)):
                u_tot_1_3[r,i,k,0]=u_tot_1_3[r,i,k,0]+uf_1_3[n,r,i,k,0] # attention car ici additionne avec main fault

for r in range(3):
    for n in range(int(round(Nombre_de_failles*(2/3)))):
        for i in range(len(tps)):
            for k in range(len(te)):
                u_tot_2_3[r,i,k,0]=u_tot_2_3[r,i,k,0]+uf_2_3[n,r,i,k,0] # attention car ici additionne avec main fault

for r in range(3):
    for n in range(int(round(Nombre_de_failles))):
        for i in range(len(tps)):
            for k in range(len(te)):
                u_tot_3_3[r,i,k,0]=u_tot_3_3[r,i,k,0]+uf_3_3[n,r,i,k,0] # attention car ici additionne avec main fault

xt=u_tot[0,:,int(round(where))-1,0]
zt=u_tot[2,:,int(round(where))-1,0]
xu_1_3=u_tot_1_3[0,:,int(round(where))-1,0]
zu_1_3=u_tot_1_3[2,:,int(round(where))-1,0]
xu_2_3=u_tot_2_3[0,:,int(round(where))-1,0]
zu_2_3=u_tot_2_3[2,:,int(round(where))-1,0]
xu_3_3=u_tot_3_3[0,:,int(round(where))-1,0]
zu_3_3=u_tot_3_3[2,:,int(round(where))-1,0]

v_tot=np.zeros((3,len(tps),len(te),len(ph)))
v_tot_1_3=np.zeros((3,len(tps),len(te),len(ph)))
v_tot_2_3=np.zeros((3,len(tps),len(te),len(ph)))
v_tot_3_3=np.zeros((3,len(tps),len(te),len(ph)))

for r in range(3):
    for n in range(int(round(Nombre_de_failles/3))):
        for i in range(len(tps)):
            for k in range(len(te)):
                v_tot_1_3[r,i,k,0]=v_tot_1_3[r,i,k,0]+vf_1_3[n,r,i,k,0] # attention car ici additionne avec main fault

for r in range(3):
    for n in range(int(round(Nombre_de_failles*(2/3)))):
        for i in range(len(tps)):
            for k in range(len(te)):
                v_tot_2_3[r,i,k,0]=v_tot_2_3[r,i,k,0]+vf_2_3[n,r,i,k,0] # attention car ici additionne avec main fault

for r in range(3):
    for n in range(int(round(Nombre_de_failles))):
        for i in range(len(tps)):
            for k in range(len(te)):
                v_tot_3_3[r,i,k,0]=v_tot_3_3[r,i,k,0]+vf_3_3[n,r,i,k,0] # attention car ici additionne avec main fault

xv_1_3=v_tot_1_3[0,:,int(round(where))-1,0]
zv_1_3=v_tot_1_3[2,:,int(round(where))-1,0]
xv_2_3=v_tot_2_3[0,:,int(round(where))-1,0]
zv_2_3=v_tot_2_3[2,:,int(round(where))-1,0]
xv_3_3=v_tot_3_3[0,:,int(round(where))-1,0]
zv_3_3=v_tot_3_3[2,:,int(round(where))-1,0]

# Visualisation du sismogramme synthétique
plt.subplot(gs[0,:2])
axes = plt.gca()
axes.set_ylim(-0.16,0.16)
ux1, = plt.plot(t,x,'k',label='$x_1$MF')
ux3, = plt.plot(t,z,'k--',label='$x_3$MF')
#upx1, = plt.plot(t,xp,'b',label='$x_1^P$MF')
#upx3, = plt.plot(t,zp,'b--',label='$x_3^P$MF')
#usx1, = plt.plot(t,xs,'g',label='$x_1^S$MF')
#usx3, = plt.plot(t,zs,'g--',label='$x_3^S$MF')
#utx1, = plt.plot(t,xt,'r',label='$x_1$tot')
#utx3, = plt.plot(t,zt,'r--',label='$x_3$tot')
ux1_1, = plt.plot(t,xu_1_3,'r',label='$x_1$MF')
ux3_1, = plt.plot(t,zu_1_3,'r--',label='$x_3$MF')
ux1_2, = plt.plot(t,xu_2_3,'b',label='$x_1$MF')
ux3_2, = plt.plot(t,zu_2_3,'b--',label='$x_3$MF')
ux1_3, = plt.plot(t,xu_3_3,'g',label='$x_1$MF')
ux3_3, = plt.plot(t,zu_3_3,'g--',label='$x_3$MF')

plt.xlabel('temps (s)')
plt.ylabel('déplacement (m)')
axes.legend(loc='center left', bbox_to_anchor=(-0.3, 0, 0.46, 0.97), framealpha=1, facecolor=axcolor, edgecolor='k')
plt.xlim(start_time-1,start_time+duration+int(round(durées[Nombre_de_failles-1]))+3)

# VÉLOCIMÈTRE
plt.subplot(gs[0,2:])
axesv = plt.gca()
vx1, = plt.plot(t,xv,'k',label='$x_1$MF')
vx3, = plt.plot(t,zv,'k--',label='$x_3$MF')
#vxt, =  plt.plot(t,vxt,'g',label='$x_1$tot')
#vzt, =  plt.plot(t,vzt,'g--',label='$x_3$tot')
vx1_1, = plt.plot(t,xv_1_3,'r',label='$x_1$MF')
vx3_1, = plt.plot(t,zv_1_3,'r--',label='$x_3$MF')
vx1_2, = plt.plot(t,xv_2_3,'b',label='$x_1$MF')
vx3_2, = plt.plot(t,zv_2_3,'b--',label='$x_3$MF')
vx1_3, = plt.plot(t,xv_3_3,'g',label='$x_1$MF')
vx3_3, = plt.plot(t,zv_3_3,'g--',label='$x_3$MF')
plt.xlabel('temps (s)')
plt.ylabel('vitesse (m/s)')
axesv.legend(loc='upper right', bbox_to_anchor=(0, 0, 1.23, 0.95), framealpha=1, facecolor=axcolor, edgecolor='k')
plt.xlim(start_time-1,start_time+duration+int(round(durées[Nombre_de_failles-1]))+1)

# Visualisation de notre "localisation"
plt.subplot(gs[2,3])
axes1 = plt.gca()
#plt.plot(zd,xd)
#plt.xlabel('x1 composante')
#plt.ylabel('x3 composante')
plt.xlim(-1,1)
plt.ylim(-1,1)
wherep=(int(round(where))/step)
t = np.linspace(0,pi*2,step)
plt.plot(np.cos(t), np.sin(t),'k',linewidth=1)
axes1.set_aspect('equal', 'box')
axes1.xaxis.set_visible(False)
axes1.yaxis.set_visible(False)
l, = plt.plot(np.cos(2*pi*wherep),np.sin(2*pi*wherep),'bD')
plt.xlabel('Localisation de la station, orientation de la faille et positions des axes')
#plt.plot([orientation_représentation(x) for x in range(4)],'k')
#a, = plt.plot(np.cos(2*pi*wherep),np.sin(2*pi*wherep),-np.cos(2*pi*wherep),-np.sin(2*pi*wherep),head_width=0.05,head_length=0.1, fc='k', ec='k')
plt.arrow(0,0,-np.cos((orientation_voulue-90)*(2*pi/360))*0.5,-np.sin((orientation_voulue-90)*(2*pi/360))*0.5,head_width=0.05,head_length=0.1, fc='k', ec='k')
plt.arrow(10,np.tan(orientation_voulue*(2*pi/360))*10,-20,np.tan(orientation_voulue*(2*pi/360))*(-20),head_width=0.05,head_length=0.1, fc='k', ec='k', lw=3,label='$x_3$')
plt.arrow(0,0,0,0.8,head_width=0.2,head_length=0.2, fc='b', ec='b', lw=4)
plt.arrow(0,0,0.8,0,head_width=0.2,head_length=0.2, fc='b', ec='b', lw=4, label='$x_1$')

# Visualisation du motif de radiation des ondes P
plt.subplot(gs[2,0])
axes2 = plt.gca()
plt.plot(up[2,:,:,0],up[0,:,:,0],'k',label='Orbitales des ondes P')
axes2.set_aspect('equal', 'box')
axes2.xaxis.set_visible(False)
axes2.yaxis.set_visible(False)
#plt.title('Orbitales des ondes P',loc='left',fontsize=8)
#plt.ylabel('Orbitales des ondes P')#, horizontalalignment='left', verticalalignment='center')
#plt.xlabel('P WAVES PATTERN')

# Visualisation du motif de radiation des ondes S
plt.subplot(gs[2,1])
axes3 = plt.gca()
plt.plot(us[0,:,:,0],us[2,:,:,0],'k')
axes3.set_aspect('equal', 'box')
axes3.xaxis.set_visible(False)
axes3.yaxis.set_visible(False)
#plt.xaxis.set_visible(False)
#plt.yaxis.set_visible(False)
#plt.xlabel('S WAVES PATTERN')

# Visualisation du tenseur de moment appliqué à la solution pour les ondes
plt.subplot(gs[3,0])
# plot de la somme des STF
#if (aléatoire==0):
#   plt.plot(tps/duration,M0g[:],'k')
#plt.plot(tps,M0gdfd[:],'b')
#elif (aléatoire==1):
#   plt.plot(tps,M0d_a[0,:],'k')
# plot de chaque STF de chaque failles
tps_v=np.zeros(len(tps))
for i in range(len(tps)):
    tps_v[i]=tps[i]/duration

plt.plot(tps,M0g[:],'k')
for n in range(int(round(Nombre_de_failles/3))):
    plt.plot(tps,M0f_1_3[n,:],'r',linewidth=0.5)
for n in range(int(round(Nombre_de_failles*(2/3)))):
    plt.plot(tps,M0f_2_3[n,:],'b',linewidth=0.5)
for n in range(Nombre_de_failles):
    plt.plot(tps,M0f_3_3[n,:],'g',linewidth=0.5)
plt.ylabel('$\dot{M}_0$')
plt.xlabel('temps (s)')
#plt.xlim(start_time-1,start_time+duration+int(round(durées[Nombre_de_failles-1]))+1)

#plot des $\ddot{M}$
plt.subplot(gs[3,1])
for n in range(Nombre_de_failles):
    plt.plot(tps,M0fd_1_3[n,:],'r--',linewidth=0.5)
    plt.plot(tps,M0fd_2_3[n,:],'b--',linewidth=0.5)
    plt.plot(tps,M0fd_3_3[n,:],'g--',linewidth=0.5)
plt.plot(tps,M0dif[:],'k--')
plt.ylabel('$\ddot{M}_0$')
plt.xlabel('temps (s)')
plt.xlim(start_time-(1/2)*duration,start_time+duration+(1/2)*duration)

# Plot interactif

# LOCALISATION$
ax_loc = plt.axes([0.22, 0.56, 0.18, 0.03], facecolor=axcolor)#, xlabel='localisation')
s_loc = Slider(ax_loc,'Position de la \n'
               'station en degrés', 0, 360, valinit=0, valstep=45)

def update(val):
    loc = s_loc.val
    l.set_xdata(np.cos(2*pi*(int(round(loc*(step/360)))/step)))
    l.set_ydata(np.sin(2*pi*(int(round(loc*(step/360)))/step)))
    #a.set_xdata(np.cos(2*pi*(int(round(loc*(step/360)))/step)))
    #a.set_ydata(np.sin(2*pi*(int(round(loc*(step/360)))/step)))
    #a.set_dxdata(-np.cos(2*pi*(int(round(loc*(step/360)))/step)))
    #a.set_dydata(-np.sin(2*pi*(int(round(loc*(step/360)))/step)))
    ux1.set_ydata(u[0,:,int(round(loc*(step/360)))-1,0])
    ux3.set_ydata(u[2,:,int(round(loc*(step/360)))-1,0])
    ux1_1.set_ydata(u_tot_1_3[0,:,int(round(loc*(step/360)))-1,0])
    ux3_1.set_ydata(u_tot_1_3[2,:,int(round(loc*(step/360)))-1,0])
    ux1_2.set_ydata(u_tot_2_3[0,:,int(round(loc*(step/360)))-1,0])
    ux3_2.set_ydata(u_tot_2_3[2,:,int(round(loc*(step/360)))-1,0])
    ux1_3.set_ydata(u_tot_3_3[0,:,int(round(loc*(step/360)))-1,0])
    ux3_3.set_ydata(u_tot_3_3[2,:,int(round(loc*(step/360)))-1,0])
    #upx1.set_ydata(up[0,:,int(round(loc*(step/360)))-1,0])
    #upx3.set_ydata(up[2,:,int(round(loc*(step/360)))-1,0])
    #usx1.set_ydata(us[0,:,int(round(loc*(step/360)))-1,0])
    #usx3.set_ydata(us[2,:,int(round(loc*(step/360)))-1,0])
    #utx1.set_ydata(u_tot[0,:,int(round(loc*(step/360)))-1,0])
    #utx3.set_ydata(u_tot[2,:,int(round(loc*(step/360)))-1,0])
    vx1.set_ydata(vd[0,:,int(round(loc*(step/360)))-1,0])
    vx3.set_ydata(vd[2,:,int(round(loc*(step/360)))-1,0])
    vx1_1.set_ydata(v_tot_1_3[0,:,int(round(loc*(step/360)))-1,0])
    vx3_1.set_ydata(v_tot_1_3[2,:,int(round(loc*(step/360)))-1,0])
    vx1_2.set_ydata(v_tot_2_3[0,:,int(round(loc*(step/360)))-1,0])
    vx3_2.set_ydata(v_tot_2_3[2,:,int(round(loc*(step/360)))-1,0])
    vx1_3.set_ydata(v_tot_3_3[0,:,int(round(loc*(step/360)))-1,0])
    vx3_3.set_ydata(v_tot_3_3[2,:,int(round(loc*(step/360)))-1,0])
    #vxt.set_ydata(v_tot[0,:,int(round(loc*(step/360)))-1,0])
    #vzt.set_ydata(v_tot[2,:,int(round(loc*(step/360)))-1,0])
    fig.canvas.draw_idle() # nécessaire pour redessiner la figure au fur et à mesure

s_loc.on_changed(update)

# Différents types de déplacements
lines=[ux1, ux3]#, upx1, upx3, usx1, usx3, utx1, utx3]#, vx1, vx3]
ax_disp = plt.axes([0.9, 0.1, 0.09, 0.285], facecolor=axcolor)
labels = [str(line.get_label()) for line in lines]
visibility = [line.get_visible() for line in lines]
check = CheckButtons(ax_disp, labels, visibility)#, active=0)
def dispfunc(label):
    index=labels.index(label)
    lines[index].set_visible(not lines[index].get_visible())
    fig.canvas.draw_idle()

check.on_clicked(dispfunc)

# Différents types de vélocimètre
linest=[vx1, vx3]#, vxt, vzt]
ax_dispt = plt.axes([0.9, 0.4, 0.09, 0.2], facecolor=axcolor)
labelst = [str(line.get_label()) for line in linest]
visibilityt = [line.get_visible() for line in linest]
checkt = CheckButtons(ax_dispt, labelst, visibilityt)#, active=0)
def dispfunct(label):
    indext=labelst.index(label)
    linest[indext].set_visible(not linest[indext].get_visible())
    fig.canvas.draw_idle()

checkt.on_clicked(dispfunct)


# Légende
plt.subplot(gs[3,2])
ax_leg = plt.gca() #plt.axes([0.52, 0.41, 0.22, 0.1], frameon=False)
ax_leg.xaxis.set_visible(False)
ax_leg.yaxis.set_visible(False)
ax_leg.annotate(' \n'
                '$Légende$ :\n'
                'Distance source - récepteur = '+ str(rayon) + 'km \n'
                'Orientation de la faille = '+ str(orientation_voulue) + '° \n'
                'Vitesse extension faille principale = '+ str(v) + 'km/s \n'
                'Début du séisme = '+ str(start_time) + 's \n'
                'Temps de durée du séisme = '+ str(duration) + 's \n'
                'Nombre de failles secondaires '+str(Nombre_de_failles) +'\n',(0,0), bbox=dict(boxstyle='round', fc=axcolor), fontsize=8)

# Titre sur subplot
ax_titre = plt.axes([0.239, 0.897, 0.1, 0.1], frameon=False)
ax_titre.xaxis.set_visible(False)
ax_titre.yaxis.set_visible(False)
ax_titre.annotate("OUTIL INTERACTIF DE VISUALISATION DE L'INFLUENCE D'UNE\n"
                  "DISTRIBUTION DE FRACTURES SUR LES RADIATIONS D'ONDES",(0.7,0.3), weight = 'bold', bbox=dict(boxstyle='round', fc='w'))

# Titre pour orbitales
#axop = plt.axes([0.1, 0.264, 0.15, 0.1], frameon=False)
#axop.xaxis.set_visible(False)
#axop.yaxis.set_visible(False)
#axop.annotate("Orbitales P",(0.7,0.3), weight = 'bold', bbox=dict(boxstyle='round', fc='w'))
axos = plt.axes([0.1, 0.264, 0.51, 0.1], frameon=False)
axos.xaxis.set_visible(False)
axos.yaxis.set_visible(False)
axos.annotate("Orbitales S",(0.7,0.3), weight = 'bold', bbox=dict(boxstyle='round', fc='w'))

#axes plan cartésien
axx1 = plt.axes([0.775, 0.704, 0.09, 0.285], frameon=False)
axx1.xaxis.set_visible(False)
axx1.yaxis.set_visible(False)
#axx1.annotate("$x_1$",(0.7,0.3), weight = 'bold')#, bbox=dict(boxstyle='round', fc='w'))
axx3 = plt.axes([0.719, 0.806, 0.09, 0.285], frameon=False)
axx3.xaxis.set_visible(False)
axx3.yaxis.set_visible(False)
#axx3.annotate("$x_3$",(0.7,0.3), weight = 'bold')#, bbox=dict(boxstyle='round', fc='w'))

# Visualisation de la zone endomagée
plt.subplot(gs[2,2])
axes4 = plt.gca()
axes4.xaxis.set_visible(False)
axes4.yaxis.set_visible(False)
def MF(x):
    return np.tan(orientation_voulue*(2*pi/360))*x + 0

ff=np.zeros(Nombre_de_failles)
def failles(x):
    return np.tan(orientation_failles[:]*(2*pi/360))*x + 0 #+ np.random.uniform(-5,5)

def position_failles():
    L=[]
    for i in range(Nombre_de_failles):
        a=np.random.randint(0,8)
        b=np.random.randint(a+1,a+4)
        L.append([i for i in range(a,b+1)])
    return L

plt.plot([MF(x) for x in range(10)],'k',lw=2)
L=position_failles()

for i in range(Nombre_de_failles):
    plt.plot(L[i],[np.tan(orientation_failles[i]*(2*pi/360))*x for x in L[i]])
axes4.set_aspect('equal', 'box')
plt.xlim(-2,12)
plt.ylim(-7,7)
plt.title('Représentation virtuelle de la \n'
          'distribution des failles',fontsize=10)

#titre pour tenseur de moment
axM0 = plt.axes([0.05, -0.013, 0.09, 0.1], frameon=False)
axM0.xaxis.set_visible(False)
axM0.yaxis.set_visible(False)
axM0.annotate("En noir : $\dot{M}_0$ faille principale     \n"
              "En couleur : $\dot{M}_0$ failles secondaires",(0.7,0.3), bbox=dict(boxstyle='round', fc=axcolor))

#gs1.tight_layout()
plt.show()
