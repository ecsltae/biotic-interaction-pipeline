"""Relation polarity lexicon.  pol=+1 -> the SUBJECT of the relation plays the
AGENT role (parasite/pathogen/predator/pollinator/vector); pol=-1 -> the subject
plays the PATIENT role (host/prey/victim/exposed); pol=0 -> symmetric.
Layer 1: ROBI concept ids (exact, free -- the pool carries the id in triplet_key).
Layer 2: morphology over the canonical relation vocabulary."""
import json, re
from pathlib import Path
_HERE = Path(__file__).resolve().parent

_M=json.load(open(_HERE / 'data' / 'robi_maps.json'))
POL_ID=_M['pol']; TERM=_M['term']; INV=_M['inv']
POL_ID['RO_0002451']=1   # transmitted by -> subject is the pathogen (high rank on the role axis)
POL_ID['RO_0002459']=-1  # is vector for  -> subject is the vector (low rank on the role axis)
POL_ID['RO_0002460']=1   # has vector     -> subject is the pathogen
POL_ID['RO_e000024']=1   # has dispersal vector
_robi=json.load(open(_HERE / 'data' / 'robiext_v2025.json'))['concepts']
SURF={}
for c in _robi:
    p=POL_ID.get(c['id'],0)
    for t in [c['preferred_term']['term']]+[s['term'] for s in c.get('synonyms',[])]:
        SURF.setdefault(t.lower().strip(),(p,c['id']))
# ROBI lists 'preys','prey on','predator','predatory' as synonyms of 'preyed upon by' (RO_0002458).
# That is an ontology error: those forms are ACTIVE. Override, documented.
for t in ['preys','prey on','preys on','predator','predator of','predators','predators of','predatory','preying','predates','predate']:
    SURF[t]=(1,'RO_0002439')
# Same class of ontology error: RO_0002445 'parasitized by' lists the ACTIVE forms
# 'parasitizing'/'parasitizes'/'parasitize' as synonyms.  Override.
for t in ['parasitizing','parasitizes','parasitize','parasitising','parasitises','parasitise','parasitizes on']:
    SURF[t]=(1,'RO_0002444')
for t in ['transmitting','transmits','transmit']:
    SURF[t]=(-1,'RO_0002459')
# AXIS NOTE.  The role axis used here is "parasite/pathogen/consumer rank".  For the
# vector relations the subject of 'transmitted by' is the PATHOGEN (high rank) and the
# object is the VECTOR (arthropod, lower rank), so on THIS axis 'transmitted by' is +1
# and 'is vector for' is -1 -- the opposite of the naive grammatical-voice reading.
for t in ['transmitted by','was transmited','had been transmited','have transmited','was transmiting by',
          'have been transmited','had transmited','was being transmited','transmit by','is transmiting by',
          'is being transmited','has transmited','transmitted']:
    SURF[t]=(1,'RO_0002451')
for t in ['is vector for','vector for','vectors for','has vector','vector of','vectors of','vector']:
    SURF[t]=(-1,'RO_0002459')
AGENT_NOUN=r'^(pathogen|parasite|parasitoid|predator|ectoparasite|endoparasite|hyperparasite|kleptoparasite|mesoparasite|symbiont|vector|epiphyte|pollinator|consumer|grazer|herbivore|infector)s?( of| for| to)?$'
AGENT_VERB=r'^(infect|prey|eat|invad|coloni[sz]|parasiti[sz]|pollinat|transmit|attack|kill|consum|graz|hunt|feed|visit|ingest|contaminat|infest|affect|bait|regulat|neutrali[sz]|attract|disperse|has )'
PATIENT=r'( by$)|^(exposed to|resistance to|resistant to|susceptible to|host|hosts|host of|host for|hosts of|hosts for|prey of|preyed|victim|reservoir)$'
SYM=r'^(interacts? with|associated with|co-?occurs? with|symbiosis|mutualism|commensalism|parasite-host|coevolv|migrates? with|adjacent|present in|found in|co-roosts? with|cooperates? with|communicates? with|copulates? with|compet(es?|ing|ition)( with)?|co-?infect\w*|mutualistic)'
ACTIVE_MORPH=re.compile(r'^(parasiti[sz]|infest|infect|coloni[sz]|invad|attack|kill|consum|graz|hunt|pollinat|predat|prey|eat|ingest|contaminat)(ing|es|e|s)?$')
def polarity(rel, ro_id=None):
    """returns (pol, source)"""
    for part in str(rel).split('|'):
        k=re.sub(r'\s+',' ',part.lower().strip())
        if k and ' by' not in k and ACTIVE_MORPH.match(k): return 1,'morph_active_participle'
    if ro_id and ro_id in POL_ID and POL_ID[ro_id]!=0: return POL_ID[ro_id],'robi_id'
    if ro_id and ro_id in POL_ID: return 0,'robi_id'
    for part in str(rel).split('|'):
        k=re.sub(r'\s+',' ',part.lower().strip())
        if not k: continue
        for v in (k, k.rstrip('s'), k+'s'):
            if v in SURF and SURF[v][0]!=0: return SURF[v][0],'robi_surface'
        if re.search(SYM,k): return 0,'morph_sym'
        if re.search(PATIENT,k): return -1,'morph_patient'
        if re.match(AGENT_NOUN,k): return 1,'morph_agent_noun'
        if re.match(AGENT_VERB,k): return 1,'morph_agent_verb'
        if k in SURF: return 0,'robi_surface_sym'
    return None,'none'


def polarity_to_index(p, n_pol):
    """Map a lexicon polarity to the direction head's embedding index, exactly as in training.

    p is +1 (agent-side subject), -1 (patient-side subject), 0 (symmetric relation) or None
    (relation not in the lexicon).
      n_pol == 2  (trained with {patient, agent}): agent -> 1, everything else -> 0
      n_pol == 3  (trained with {patient, agent, symmetric}): symmetric -> 2, agent -> 1, else 0
    Both mappings are the ones the training scripts used; a mismatch here is train/serve skew.
    """
    if n_pol == 3 and p == 0:
        return 2
    return 1 if (p is not None and p > 0) else 0


def is_symmetric(p):
    """A relation the lexicon calls symmetric has no subject: the interaction is bidirectional."""
    return p == 0
