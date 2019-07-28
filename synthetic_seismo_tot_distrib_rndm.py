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
ou=45 # Où se localise t-on au niveau du recépteur (à r) afin d'observer l'arrivée des ondes; en degrés.
orientation_voulue = 0 # orientation en degré
Nombre_de_failles = 30 # nombre de failles autour de la faille principale
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
alpha=np.sqrt((lambd+2*mu)/rho) # Vitesse  théorique des ondes P
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
temps_de_départ=np.zeros(Nombre_de_failles) # temps (donc position) à laquelle faille (STF) se déclanche
durées=np.zeros(Nombre_de_failles) # durée aléatoire du STF de la faille secondaire concernée
for i in range(Nombre_de_failles):
    #positions_failles[i]=np.random.uniform(v*(start_time+start_time*0.07),v*(starting_time[Extension_de_la_faille-1]+duration))# si plusieurs gaussiennes d'affilées pour définir faille principale
    positions_failles[i]=np.random.uniform(v*(start_time+start_time*0.07),v*(start_time+duration)) # ce qu'on utilise depuis dérivée effectuable sur une seule gaussienne
    temps_de_départ[i]=positions_failles[i]/v
    durées[i]=np.random.uniform(1/10*duration,1/3*duration) # Durée de la STF arbitraire, étant sécondaires, elles ont étés définies avec des longueurs allant de 1/10 à 1/3 de la durée de la faille principale
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

M0f=np.zeros((Nombre_de_failles,len(tps)))
M0fd=np.zeros((Nombre_de_failles,len(tps)))
M0pf=np.zeros((Nombre_de_failles,len(tps)))
M0df=np.zeros((Nombre_de_failles,len(tps)))
M0dfd=np.zeros((Nombre_de_failles,len(tps)))
M0gf=np.zeros((Nombre_de_failles,len(tps)))
M0gfd=np.zeros((Nombre_de_failles,len(tps)))

M0_af=np.zeros((Nombre_de_failles,len(tps)))
M0p_af=np.zeros((Nombre_de_failles,len(tps)))
M0d_af=np.zeros((Nombre_de_failles,len(tps)))
M0g_af=np.zeros((Nombre_de_failles,len(tps)))

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

M0gf=Gaussianf(M0f)

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

M0g_af=Gaussian_af(M0_af)

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

@njit()
def Gaussianfd(M0fd): # Fonction Gaussienne
    for n in range(Nombre_de_failles):
        atténuation_aléatoire=np.random.uniform(0.1*10**18,1.58*10**18) # ICI QUE FACTEUR ATTÉNUATION BIEN PLACÉ !!!!!!!!!!!!!!!!!!!!!!!!!
        for i in range(len(tps)):
            if (temps_de_départ[n]<=tps[i]<=(temps_de_départ[n]+durées[n])):
                mu=(temps_de_départ[n]+temps_de_départ[n]+durées[n])/2 # moyenne
                e=((durées[n])/(2.2*durées[n]))#+n*0.25 # écart-type + facteur d'atténuation : n*XXX #### ATTENTION PARAMÈTRE DANGEREUX !!!! #### normalement c'est bon car proportionnalité liée à vecteur durée...
                M0fd[n,i]=((tps[i]-mu)/(e**2)) * (1 / (((e*np.sqrt(2*pi)))) * np.exp((-1/2)*(((tps[i]-mu)/e)**2)))*((2*10**18)-atténuation_aléatoire)
    return M0fd

#M0gfd=Gaussianfd(M0fd) #????

#DÉRIVÉE => pour vélocimètre
################################################################################

if (aléatoire==0):
    if (gauss==1):
        # MMF=M0d
        M0df=Gaussianf(M0f)
        M0dfd=Gaussianfd(M0fd)
        for n in range(Nombre_de_failles):
            for i in range(len(tps)):
                if (i >= r/alpha):
                    M0af[n,i]=M0df[n,i-int(round(r/alpha))]
                    M0afd[n,i]=M0dfd[n,i-int(round(r/alpha))]
                    M0af_tot[i]=M0af_tot[i]+M0af[n,i] # ????
        # MMF[i]=MMF[i]+M0df[n,i] #big M0 = sum de toutes les failles
        for n in range(Nombre_de_failles):
            for i in range(len(tps)):
                if (i >= r/beta):
                    M0bf[n,i]=M0df[n,i-int(round(r/beta))]
                    M0bfd[n,i]=M0dfd[n,i-int(round(r/beta))]
                    M0bf_tot[i]=M0bf_tot[i]+M0bf[n,i]
        #    MMF[i]=MMF[i]+M0df[n,i]
    elif (gauss==0):
        #  MMF=M0d
        M0df=triangle_difff(M0pf)
        for n in range(Nombre_de_failles):
            for i in range(len(tps)):
                if (i >= r/alpha):
                    M0af[n,i]=M0df[n,i-int(round(r/alpha))]
                    M0af_tot[i]=M0af_tot[i]+M0af[n,i]
        #           MMF[i]=MMF[i]+M0df[n,i]
        for n in range(Nombre_de_failles):
            for i in range(len(tps)):
                if (i >= r/beta):
                    M0bf[n,i]=M0df[n,i-int(round(r/beta))]
                    M0bf_tot[i]=M0bf_tot[i]+M0bf[n,i]
#            MMF[i]=MMF[i]+M0df[n,i]
elif(aléatoire==1):
    if (gauss==1):
        # MMF=M0d_a
        M0d_af=Gaussian_af(M0_af)
        for n in range(Nombre_de_failles):
            for i in range(len(tps)):
                if (i >= r/alpha):
                    M0af[n,i]=M0d_af[n,i-int(round(r/alpha))]
                    M0af_tot[i]=M0af_tot[i]+M0af[n,i]
        #          MMF[i]=MMF[i]+M0d_af[n,i]
        for n in range(Nombre_de_failles):
            for i in range(len(tps)):
                if (i >= r/beta):
                    M0bf[n,i]=M0d_af[n,i-int(round(r/beta))]
                    M0bf_tot[i]=M0bf_tot[i]+M0bf[n,i]
        #            MMF[i]=MMF[i]+M0d_af[n,i]
    elif (gauss==0):
        # MMF=M0d_a
        M0d_af=triangle_diff_af(M0p_af)
        for n in range(Nombre_de_failles):
            for i in range(len(tps)):
                if (i >= r/alpha):
                    M0af[n,i]=M0d_af[n,i-int(round(r/alpha))]
                    M0af_tot[i]=M0af_tot[i]+M0af[n,i]
        #            MMF[i]=MMF[i]+M0d_af[n,i]
        for n in range(Nombre_de_failles):
            for i in range(len(tps)):
                if (i >= r/beta):
                    M0bf[n,i]=M0d_af[n,i-int(round(r/beta))]
                    M0bf_tot[i]=M0bf_tot[i]+M0bf[n,i]
#          MMF[i]=MMF[i]+M0d_af[n,i]

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

uf=np.zeros((Nombre_de_failles,3,len(tps),len(te),len(ph))) # pour failles secondaires

@njit()
def sismo_total_f(uf, rdf, tedf, phidf, rf, tef, phf, M0af, M0bf): # Solution pour le déplacement total
    for w in range(Nombre_de_failles):
        for d in range(3):
            for t in range(len(tps)):
                for i in range(len(te)):
                    for n in range(len(ph)):
                        uf[w,d,t,i,n] = (1 / (4*pi*rho*alpha**3)) * (np.sin(2*tef[w,i])*np.cos(phf[w,n])*rdf[w,d,i,n]) * (1/rf[w]) * M0af[w,t] + (1 / (4*pi*rho*beta**3)) * (np.cos(2*tef[w,i])*np.cos(phf[w,n])*tedf[w,d,i,n] - np.cos(tef[w,i])*np.sin(phf[w,n])*phidf[w,d,i,n]) * (1/rf[w]) * M0bf[w,t]
    return uf

uf=sismo_total_f(uf, rdf, tedf, phidf, rf, tef, phf, M0af, M0bf)

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

vp=np.zeros((3,len(tps),len(te),len(ph))) # Matrice de déplacement pour les ondes P
vs=np.zeros((3,len(tps),len(te),len(ph))) # Matrice de déplacement pour les ondes S

@njit()
def sismo_Pd(v, rd, ted, phid, r, te, phi, M0ad): # Solution pour le déplacement des ondes P
    for d in range(3):
        for t in range(len(tps)):
            for i in range(len(te)):
                for n in range(len(ph)):
                    u[d,t,i,n] = (1 / (4*pi*rho*alpha**3)) * (np.sin(2*te[i])*np.cos(phi[n])*rd[d,i,n]) * (1/r) * M0ad[t]
    return u

@njit()
def sismo_Sd(v, rd, ted, phid, r, te, phi, M0bd): # Solution pour le déplacement des ondes S
    for d in range(3):
        for t in range(len(tps)):
            for i in range(len(te)):
                for n in range(len(ph)):
                    v[d,t,i,n] = (1 / (4*pi*rho*beta**3)) * (np.cos(2*te[i])*np.cos(phi[n])*ted[d,i,n] - np.cos(te[i])*np.sin(phi[n])*phid[d,i,n]) * (1/r) * M0bd[t]
    return v

################################################################################
#DÉRIVÉE pour ensemlbe failles

vfd=np.zeros((Nombre_de_failles,3,len(tps),len(te),len(ph))) # pour failles secondaires

@njit()
def sismo_total_fd(vfd, rdf, tedf, phidf, rf, tef, phf, M0afd, M0bfd): # Solution pour le déplacement total
    for w in range(Nombre_de_failles):
        for d in range(3):
            for t in range(len(tps)):
                for i in range(len(te)):
                    for n in range(len(ph)):
                        vfd[w,d,t,i,n] = (1 / (4*pi*rho*alpha**3)) * (np.sin(2*tef[w,i])*np.cos(phf[w,n])*rdf[w,d,i,n]) * (1/rf[w]) * M0afd[w,t] + (1 / (4*pi*rho*beta**3)) * (np.cos(2*tef[w,i])*np.cos(phf[w,n])*tedf[w,d,i,n] - np.cos(tef[w,i])*np.sin(phf[w,n])*phidf[w,d,i,n]) * (1/rf[w]) * M0bfd[w,t]
    return vfd

vfd=sismo_total_fd(vfd, rdf, tedf, phidf, rf, tef, phf, M0afd, M0bfd)


################################################################################
# Remplissage des matrices :

u=sismo_total(u, rd, ted, phid, r, te, ph, M0a, M0b)
vd=sismo_totald(vd, rd, ted, phid, r, te, ph, M0ad, M0bd)
up=sismo_P(up, rd, ted, phid, r, te, ph, M0a)
us=sismo_S(us, rd, ted, phid, r, te, ph, M0b)
vp=sismo_P(vp, rd, ted, phid, r, te, ph, M0ad)
vs=sismo_S(vs, rd, ted, phid, r, te, ph, M0bd)

################################################################################
################################################################################
# Réalisation des figures :
################################################################################
################################################################################

where=orientation_voulue*((step-1)/360)
axcolor = 'silver' #'lightgreen'

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
vxp=vp[0,:,int(round(where))-1,0]
vzp=vp[2,:,int(round(where))-1,0]
vxs=vs[0,:,int(round(where))-1,0]
vzs=vs[2,:,int(round(where))-1,0]
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

for r in range(3):
    for i in range(len(tps)):
        for n in range(len(te)):
            u_tot[r,i,n,0]=u[r,i,n,0]

for r in range(3):
    for n in range(Nombre_de_failles):
        for i in range(len(tps)):
            for k in range(len(te)):
                u_tot[r,i,k,0]=u_tot[r,i,k,0]+uf[n,r,i,k,0]

xt=u_tot[0,:,int(round(where))-1,0]
zt=u_tot[2,:,int(round(where))-1,0]

v_tot=np.zeros((3,len(tps),len(te),len(ph)))

for r in range(3):
    for i in range(len(tps)):
        for n in range(len(te)):
            v_tot[r,i,n,0]=vd[r,i,n,0]

for r in range(3):
    for n in range(Nombre_de_failles):
        for i in range(len(tps)):
            for k in range(len(te)):
                v_tot[r,i,k,0]=v_tot[r,i,k,0]+vfd[n,r,i,k,0]

vxt=v_tot[0,:,int(round(where))-1,0]
vzt=v_tot[2,:,int(round(where))-1,0]

# Visualisation du sismogramme synthétique
plt.subplot(gs[0,:2])
plt.tight_layout()
axes = plt.gca()
axes.set_ylim(-0.16,0.16)
ux1, = plt.plot(t,x,'y',label='U$x_1$MF')
ux3, = plt.plot(t,z,'y--',label='U$x_3$MF')
upx1, = plt.plot(t,xp,'k',linewidth=1.5,label='U$x_1^P$MF')
upx3, = plt.plot(t,zp,'k--',linewidth=1.5,label='U$x_3^P$MF')
usx1, = plt.plot(t,xs,'b',linewidth=1.5,label='U$x_1^S$MF')
usx3, = plt.plot(t,zs,'b--',linewidth=1.5,label='U$x_3^S$MF')
utx1, = plt.plot(t,xt,'r',label='U$x_1$tot')
utx3, = plt.plot(t,zt,'r--',label='U$x_3$tot')

plt.xlabel('temps (s)')
plt.ylabel('déplacement (m)')
#axes.legend(loc='upper right', bbox_to_anchor=(0, 0, 1.23, 0.95), framealpha=1, facecolor=axcolor, edgecolor='k')
#axes.legend(loc='center left', bbox_to_anchor=(-0.3, 0, 0.46, 0.97), framealpha=1, facecolor=axcolor, edgecolor='k')
#axes.legend(loc='center left', framealpha=1, facecolor=axcolor, edgecolor='k')
plt.xlim(start_time-1,start_time+duration+durées[int(round(Nombre_de_failles-1))]+4)

# VÉLOCIMÈTRE
plt.subplot(gs[0,2:])
plt.tight_layout()
axesv = plt.gca()
vx1, = plt.plot(t,xv,'y',label='V$x_1$MF')
vx3, = plt.plot(t,zv,'y--',label='V$x_3$MF')
vpx1, = plt.plot(t,vxp,'k',linewidth=1.5,label='V$x_1^P$MF')
vpx3, = plt.plot(t,vzp,'k--',linewidth=1.5,label='V$x_3^P$MF')
vsx1, = plt.plot(t,vxs,'b',linewidth=1.5,label='V$x_1^S$MF')
vsx3, = plt.plot(t,vzs,'b--',linewidth=1.5,label='V$x_3^S$MF')
vxt, =  plt.plot(t,vxt,'r',label='V$x_1$tot')
vzt, =  plt.plot(t,vzt,'r--',label='V$x_3$tot')
plt.xlabel('temps (s)')
plt.ylabel('vitesse (m/s)')
#axesv.legend(loc='upper right', bbox_to_anchor=(0, 0, 1, 0.95), framealpha=1, facecolor=axcolor, edgecolor='k')
#axesv.legend(loc='upper right', framealpha=1, facecolor=axcolor, edgecolor='k')
#axesv.legend(loc='center left', bbox_to_anchor=(-0.3, 0, 0.46, 0.97), framealpha=1, facecolor=axcolor, edgecolor='k')
plt.xlim(start_time-1,start_time+duration+durées[int(round(Nombre_de_failles-1))]+4)

# Visualisation de notre "localisation"
plt.subplot(gs[2,2])
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
#plt.plot([orientation_représentation(x) for x in range(4)],'k')
#a, = plt.plot(np.cos(2*pi*wherep),np.sin(2*pi*wherep),-np.cos(2*pi*wherep),-np.sin(2*pi*wherep),head_width=0.05,head_length=0.1, fc='k', ec='k')
plt.arrow(0,0,-np.cos((orientation_voulue-90)*(2*pi/360))*0.5,-np.sin((orientation_voulue-90)*(2*pi/360))*0.5,head_width=0.05,head_length=0.1, fc='k', ec='k')
plt.arrow(10,np.tan(orientation_voulue*(2*pi/360))*10,-20,np.tan(orientation_voulue*(2*pi/360))*(-20),head_width=0.05,head_length=0.1, fc='k', ec='k', lw=3,label='$x_3$')
plt.arrow(0,0,0,0.8,head_width=0.2,head_length=0.2, fc='b', ec='b', lw=4)
plt.arrow(0,0,0.8,0,head_width=0.2,head_length=0.2, fc='b', ec='b', lw=4, label='$x_1$')
plt.title('Localisation de la station \n'
          'orientation de la faille et positions des axes',fontsize=10)

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
plt.title('Orbitales P',fontsize=10)

# Visualisation du motif de radiation des ondes S
plt.subplot(gs[2,1])
axes3 = plt.gca()
plt.plot(us[2,:,:,0],us[0,:,:,0],'k')
axes3.set_aspect('equal', 'box')
axes3.xaxis.set_visible(False)
axes3.yaxis.set_visible(False)
#plt.xaxis.set_visible(False)
#plt.yaxis.set_visible(False)
#plt.xlabel('S WAVES PATTERN')
plt.title('Orbitales S',fontsize=10)

# Visualisation du tenseur de moment appliqué à la solution pour les ondes
plt.subplot(gs[3,0])
axesm = plt.gca()
# plot de la somme des STF
if (aléatoire==0):
    plt.plot(tps/duration,M0g[:],'k')
#plt.plot(tps,M0gdfd[:],'b')
elif (aléatoire==1):
   plt.plot(tps,M0d_a[0,:],'k')
# plot de chaque STF de chaque failles
tps_v=np.zeros(len(tps))
for i in range(len(tps)):
    tps_v[i]=tps[i]/(duration)#+0.3*duration)

if (aléatoire==0):
    plt.plot(tps,M0g[:],'k')
    for n in range(Nombre_de_failles): # failles secondaires
        plt.plot(tps,M0df[n,:],linewidth=0.5)
plt.ylabel('$\dot{M}_0$')
plt.xlabel('temps (s)')
plt.xlim(start_time-1,start_time+duration+durées[int(round(Nombre_de_failles-1))]+4)
plt.title('Derivée du tenseur de moment \n'
          'En noir MF et en couleur FS',fontsize=10)
#axesm.xaxis.set_visible(False)
#axesm.yaxis.set_visible(False)

#plot des $\ddot{M}$
plt.subplot(gs[3,1])
axesmd = plt.gca()
if (aléatoire==0):
    for n in range(Nombre_de_failles):
        plt.plot(tps,M0dfd[n,:],'--',linewidth=0.5)
plt.plot(tps,M0dif[:],'k--')
plt.ylabel('$\ddot{M}_0$')
plt.xlabel('temps (s)')
plt.xlim(start_time-1,start_time+duration+durées[int(round(Nombre_de_failles-1))]+4)
plt.title('Derivée seconde du tenseur de moment \n'
          'En noir MF et en couleur FS',fontsize=10)
#axesmd.xaxis.set_visible(False)
#axesmd.yaxis.set_visible(False)

# Plot interactif

# LOCALISATION$
ax_loc = plt.axes([0.45, 0.6, 0.18, 0.03], facecolor=axcolor)#, xlabel='localisation')
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
    upx1.set_ydata(up[0,:,int(round(loc*(step/360)))-1,0])
    upx3.set_ydata(up[2,:,int(round(loc*(step/360)))-1,0])
    usx1.set_ydata(us[0,:,int(round(loc*(step/360)))-1,0])
    usx3.set_ydata(us[2,:,int(round(loc*(step/360)))-1,0])
    utx1.set_ydata(u_tot[0,:,int(round(loc*(step/360)))-1,0])
    utx3.set_ydata(u_tot[2,:,int(round(loc*(step/360)))-1,0])
    vx1.set_ydata(vd[0,:,int(round(loc*(step/360)))-1,0])
    vx3.set_ydata(vd[2,:,int(round(loc*(step/360)))-1,0])
    vpx1.set_ydata(vp[0,:,int(round(loc*(step/360)))-1,0])
    vpx3.set_ydata(vp[2,:,int(round(loc*(step/360)))-1,0])
    vsx1.set_ydata(vs[0,:,int(round(loc*(step/360)))-1,0])
    vsx3.set_ydata(vs[2,:,int(round(loc*(step/360)))-1,0])
    vxt.set_ydata(v_tot[0,:,int(round(loc*(step/360)))-1,0])
    vzt.set_ydata(v_tot[2,:,int(round(loc*(step/360)))-1,0])
    fig.canvas.draw_idle() # nécessaire pour redessiner la figure au fur et à mesure

s_loc.on_changed(update)

# Différents types de déplacements
lines=[ux1, ux3, upx1, upx3, usx1, usx3, utx1, utx3]#, vx1, vx3]
ax_disp = plt.axes([0.8, 0.4, 0.09, 0.285], facecolor=axcolor)
labels = [str(line.get_label()) for line in lines]
visibility = [line.get_visible() for line in lines]
check = CheckButtons(ax_disp, labels, visibility)#, active=0)
def dispfunc(label):
    index=labels.index(label)
    lines[index].set_visible(not lines[index].get_visible())
    fig.canvas.draw_idle()

check.on_clicked(dispfunc)

# Différents types de vélocimètre
linest=[vx1, vx3, vpx1, vpx3, vsx1, vsx3, vxt, vzt]
ax_dispt = plt.axes([0.9, 0.4, 0.09, 0.285], facecolor=axcolor)
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
ax_leg.annotate('$Légende$ :\n'
                'Distance source - récepteur = '+ str(rayon) + 'km \n'
                'Orientation de la faille = '+ str(orientation_voulue) + '° \n'
                'Vitesse extension faille principale = '+ str(v) + 'km/s \n'
                'Début du séisme = '+ str(start_time) + 's \n'
                'Temps de durée du séisme = '+ str(duration) + 's \n'
                'Nombre de failles secondaires '+str(Nombre_de_failles) +'\n'
                'MF = faille principale ; FS = failles secondaires',(0,0), bbox=dict(boxstyle='round', fc=axcolor), fontsize=12)

# Titre sur subplot
ax_titre = plt.axes([0.249, 0.918, 0.1, 0.1], frameon=False)
ax_titre.xaxis.set_visible(False)
ax_titre.yaxis.set_visible(False)
ax_titre.annotate("OUTIL INTERACTIF DE VISUALISATION DE L'INFLUENCE D'UNE\n"
                  "DISTRIBUTION DE FRACTURES SUR LES RADIATIONS D'ONDES",(0.7,0.3), weight = 'bold', bbox=dict(boxstyle='round', fc='silver'))

# Titre pour orbitales
axop = plt.axes([0.1, 0.48, 0.15, 0.1], frameon=False)
axop.xaxis.set_visible(False)
axop.yaxis.set_visible(False)
#axop.annotate("Orbitales P",(0.7,0.3), weight = 'bold', bbox=dict(boxstyle='round', fc='w'))
axos = plt.axes([0.1, 0.48, 0.3, 0.1], frameon=False)
axos.xaxis.set_visible(False)
axos.yaxis.set_visible(False)
#axos.annotate("Orbitales S",(0.7,0.3), weight = 'bold', bbox=dict(boxstyle='round', fc='w'))

#axes plan cartésien
axx1 = plt.axes([0.635, 0.262, 0.09, 0.285], frameon=False)
axx1.xaxis.set_visible(False)
axx1.yaxis.set_visible(False)
axx1.annotate("$x_1$",(0.7,0.3), weight = 'bold')#, bbox=dict(boxstyle='round', fc='w'))
axx3 = plt.axes([0.6, 0.309, 0.09, 0.285], frameon=False)
axx3.xaxis.set_visible(False)
axx3.yaxis.set_visible(False)
axx3.annotate("$x_3$",(0.7,0.3), weight = 'bold')#, bbox=dict(boxstyle='round', fc='w'))

# Visualisation de la zone endomagée
#plt.subplot(gs[2,2])
#axes4 = plt.gca()
#axes4.xaxis.set_visible(False)
#axes4.yaxis.set_visible(False)
#def MF(x):
#  return np.tan(orientation_voulue*(2*pi/360))*x + 0

#ff=np.zeros(Nombre_de_failles)
#def failles(x):
#    return np.tan(orientation_failles[:]*(2*pi/360))*x + 0 #+ np.random.uniform(-5,5)

#def position_failles():
#    L=[]
        #    for i in range(Nombre_de_failles):
        #        a=np.random.randint(0,8)
        #        b=np.random.randint(a+1,a+4)
        #        L.append([i for i in range(a,b+1)])
#    return L

#plt.plot([MF(x) for x in range(10)],'k',lw=2)
#L=position_failles()

#for i in range(Nombre_de_failles):
#    plt.plot(L[i],[np.tan(orientation_failles[i]*(2*pi/360))*x for x in L[i]])
#axes4.set_aspect('equal', 'box')
#plt.xlim(-2,12)#
#plt.ylim(-7,7)
    #plt.title('Représentation virtuelle de la \n'
#         'distribution des failles',fontsize=10)

#titre pour tenseur de moment
#axM0 = plt.axes([0.05, -0.013, 0.09, 0.1], frameon=False)
#axM0.xaxis.set_visible(False)
#axM0.yaxis.set_visible(False)
#axM0.annotate("En noir : $\dot{M}_0$ faille principale     \n"
#              "En couleur : $\dot{M}_0$ failles secondaires",(0.7,0.3), bbox=dict(boxstyle='round', fc=axcolor))

#gs1.tight_layout()
plt.show()
