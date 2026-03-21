11/02 : 
Repo installation
putting pragma for opm around main loops

18/02 : 
Gonna try to use vector and xsimd
Saw a square root in the code. Gonna try to use the fast inverse square root algorithm

4/03:
Trucs à faire : 
SIMD : 
Checker les options de compilation (e.g. FAST_MATH)
Côté mémoire : Calcul des distances 2 fois 

Côté calcul : 
Checker la réciproque de la racine carré 

Changer le if / max en masque / opérations booléennes vectorielles

Essaie d'une implémentation en SIMD : grosse divergence au début.

11/03 : 
Source de l'erreur : mauvais chargement des particules j 
Base fonctionne : environ 60FPS en validation, error min de l'ordre de 1e-6

175 fps -> 54 fps

16/03 : 
Implemented fma : +2/4 fps 

Details : 2 masks (+ check self) aroung 54 fps

17/03 :
Loaded magic numbers outside of loops, used xsimd math functions explicitely.
Implemnted fma + other 
189 fps init -> 62 / 64 fps après quelques itéarations (stabilisation autour de 67)


GPU : 
BAse : 
90 FPS - 40 FPS

Naive with better write : 
500FPS - 245 FPS.

Usage of fmaf : 
Or 1000 FPS -> 500 FPS

Float4 : No increase

Tile and thread blocks : 1600 FPS - 850 FPS

CPU : 
Using other batch loading makes it slower (prob. cause of more memory access + rotate operations on batches)

Replacing mask with a max yiels better results -> 200 FPS - 67 FPS 