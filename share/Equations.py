import numpy as np
from scipy.integrate import solve_ivp
import sdeint


class NetworkEquations:
    def __init__(self):
        self.array_of_elements = None
        self.n = None
        self.out_of_network_arguments = None
        self.out_of_network_names = None
        self.Nodes()
        self.networkparameters()
    def Nodes(self):
        self.array_of_elements = np.array(['IFNGR', 'CSF2RA', 'IL1R', 'TLR4', 'FCGR', 'IL4RA', 'IL10R', 'STAT1', 'MSTAT5', 'MNFKB',
                                           'PPARG', 'STAT6', 'JMJD3', 'STAT3', 'SOCS3', 'IRF3', 'ERK', 'KLF4', 'SOCS1', 'IRF4',
                                           'IL1BOUT', 'IL12OUT', 'TAK123', 'JNKMAPK', 'ERK12', 'MSK12', 'GSK3B', 'IL10OUT', 'AKT1', 'AKT3',
                                           'TLR2', 'TLR3', 'TLR7', 'TLR8', 'TLR9', 'MYD88', 'IRAK4', 'IRAK12', 'TRAF6', 'TAB23TAK',
                                           'MKK36', 'P38', 'CREB1', 'MKK47', 'JNK', 'CJUN', 'MKK12', 'CFOS', 'MAP1', 'NEMOIKKB',
                                           'RIP1', 'TLR4END', 'TRIF', 'TRAF3', 'TBK1IKKI', 'IKKA', 'IRF7', 'IL6OUT', 'IL18OUT', 'IL33OUT',
                                           'IFNA', 'PHAGOCYTOSIS', 'PHAGOSOME', 'PROCESSING', 'MHC2', 'CD8086', 'ITAM', 'SYK', 'CARD9', 'VAV',
                                           'RAC', 'CDC42', 'WASP', 'WAVE', 'MCA', 'LYN', 'FYN', 'DAP12', 'MPI3K', 'MPLC',
                                           'MAKT', 'MPKC', 'CD11BCD18', 'HCK', 'FGR', 'FCAR', 'MMTOR', 'MMTORC1', 'MMTORC2', 'MLKB1',
                                           'MAMPK', 'MGlycolysis', 'MOXPHOS', 'MAMPATPratio', 'MHIF1A', 'RIG1', 'MAVS', 'TRADD', 'NEMOIKKAB','NEMOTBK1IKKE',
                                           'TNFR1', 'TRAF2', 'TNFAOUT', 'TCR', 'CD28', 'AP1', 'CD25', 'IL2G', 'IL2E', 'MTOR',
                                           'ZAP70', 'STAT5', 'NFAT', 'NFKB', 'AKT', 'CTLA4', 'CTLA4DIM', 'BCL2', 'NDRG1', 'DAG',
                                           'SOS', 'RASGTPR', 'LCK', 'PDK1', 'LAT', 'PLC', 'PI3K', 'PIP2', 'PIP3', 'IP3',
                                           'CA', 'PKC', 'TBET', 'IFNG', 'GATA3', 'IL4', 'FOXP3', 'IL10', 'TGFB','RORGT',
                                           'IL21', 'IL17', 'BCL6', 'IL9', 'CD40L', 'MTORC1', 'MTORC2', 'LKB1', 'AMPK','Glycolysis',
                                           'GLUTAMINOLISIS', 'AKG', 'OXPHOS', 'AMPATPratio', 'HIF1A'])
        self.n = len(self.array_of_elements)
        self.initial_conditions=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0])
        return self.array_of_elements

    def networkparameters(self):
        self.out_of_network_arguments = np.array(['b', 'LP', 'DSRNA', 'SSRNA', 'CPGDNA', 'IFNGE', 'IFNGL', 'MHC2E', 'CD8086E', 'GMCSFE',
                                                  'IL1BE', 'LPSE', 'IL4E', 'IL4L', 'TGFBE', 'IL10E', 'IL10L', 'IL12E', 'IC3B', 'IGGC',
                                                  'IGAC', 'IL6E', 'IL21E', 'IL18E', 'IL33E', 'MGLC', 'GLC', 'CITDSRNA', 'CITSSRNA', 'TNFA',
                                                  'CIAP', 'GLN', 'MFA', 'FA', 'TRP', 'MO2', 'O2', 'METF', 'RAPA', 'PRED',
                                                  'CS', 'TCD8086', 'TMHC2'])
        self.out_of_network_names = np.array(['Saturation rate', 'Lipoproteins (LP)', 'Double-stranded RNA (dsRNA)', 'Single-stranded RNA (ssRNA)',
                                              'CpG DNA (unmethylated bacterial DNA)', 'Interferon gamma (external)', 'Interferon gamma (local)', 'MHC class II (external)',
                                              'CD80/CD86 (external)', 'Granulocyte-macrophage colony-stimulating factor (GM-CSF)', 'Interleukin-1 beta (external)',
                                              'LPS (Lipopolysaccharide)', 'Interleukin-4 (external)', 'Interleukin-4 (local)', 'Transforming growth factor beta (TGF-β)',
                                              'Interleukin-10 (external)', 'Interleukin-10 (local)', 'Interleukin-12 (local)', 'Complement component iC3b',
                                              'Immunoglobulin G (IgG immune complexes)', 'Immunoglobulin A (IgA immune complexes)', 'Interleukin-6 (local)',
                                              'Interleukin-21 (local)', 'Interleukin-18 (local)', 'Interleukin-33 (local)', 'Glucose (macrophage)', 'Glucose',
                                              'Cytosolic double-stranded RNA', 'Cytosolic single-stranded RNA', 'Tumor necrosis factor alpha (TNF-α)',
                                              'Cellular inhibitors of apoptosis proteins (cIAP)', 'Glutamine', 'Macrophage fatty acids', 'Fatty acids', 'Tryptophan',
                                              'Molecular oxygen (macrophage)', 'Molecular oxygen', 'Methionine flux', 'Rapamycin (mTOR inhibitor)',
                                              'Prednisone (glucocorticoid)', 'Cholesterol synthesis / cellular sterols', 'CD28-CD80/CD86 interaction time', 'TCR-MHC II interaction time'])
        self.initial_out=np.array([8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 1.0, 0.0, 0.0, 0.0, 0.0, 1, 1, 1, 1, 1, 1, 0.0, 0.0, 0.0, 0.0, 30, 20])
        return self.out_of_network_arguments

    @staticmethod
    def interaction_terms(t, q, p, w, exp=np.exp):
        """Fill `w` with the 155 probabilistic-logic terms w_i(t, q, p).

        `w` is any container with item assignment: `np.zeros(n)` for the
        numeric ODE, or `[0] * n` with sympy symbols (and `exp=sympy.exp`)
        to build the symbolic Jacobian used by the sensitivity analysis.
        """
        w[0] = (p[5]+q[133]-p[5]*q[133])
        w[1] = p[9]
        w[2] = (p[10]+q[20]-p[10]*q[20])
        w[3] = p[11]
        w[4] = ((p[19]+p[19]*p[10]-p[19]*p[19]*p[10])+q[30]-(p[19]+p[19]*p[10]-p[19]*p[19]*p[10])*q[30])
        w[5] = (p[12]+q[135]-p[12]*q[135])
        w[6] = ((p[15]+q[27]-p[15]*q[27])+q[137]-(p[15]+q[27]-p[15]*q[27])*q[137])
        w[7] = q[0]*(1-(q[18]+q[13]-q[18]*q[13]))
        w[8] = q[1]*(1-((q[13]+q[19]-q[13]*q[19])+q[18]-(q[13]+q[19]-q[13]*q[19])*q[18]))
        w[9] = (((((q[2]+q[49]-q[2]*q[49])+q[68]-(q[2]+q[49]-q[2]*q[49])*q[68])+q[80]-((q[2]+q[49]-q[2]*q[49])+q[68]-(q[2]+q[49]-q[2]*q[49])*q[68])*q[80])+q[81]-(((q[2]+q[49]-q[2]*q[49])+q[68]-(q[2]+q[49]-q[2]*q[49])*q[68])+q[80]-((q[2]+q[49]-q[2]*q[49])+q[68]-(q[2]+q[49]-q[2]*q[49])*q[68])*q[80])*q[81])+q[50]*q[101]*p[30]-((((q[2]+q[49]-q[2]*q[49])+q[68]-(q[2]+q[49]-q[2]*q[49])*q[68])+q[80]-((q[2]+q[49]-q[2]*q[49])+q[68]-(q[2]+q[49]-q[2]*q[49])*q[68])*q[80])+q[81]-(((q[2]+q[49]-q[2]*q[49])+q[68]-(q[2]+q[49]-q[2]*q[49])*q[68])+q[80]-((q[2]+q[49]-q[2]*q[49])+q[68]-(q[2]+q[49]-q[2]*q[49])*q[68])*q[80])*q[81])*q[50]*q[101]*p[30])*(1-(((q[13]+q[10]-q[13]*q[10])+q[17]-(q[13]+q[10]-q[13]*q[10])*q[17])+q[42]-((q[13]+q[10]-q[13]*q[10])+q[17]-(q[13]+q[10]-q[13]*q[10])*q[17])*q[42]))
        w[10] = q[5]
        w[11] = q[5]*(1-q[7])
        w[12] = q[5]
        w[13] = (q[6]+q[16]*q[41]-q[6]*q[16]*q[41])*(1-(q[10]+q[14]-q[10]*q[14]))
        w[14] = (q[9]+q[7]-q[9]*q[7])
        w[15] = (q[54]+q[99]-q[54]*q[99])
        w[16] = ((q[4]+q[46]-q[4]*q[46])+q[78]-(q[4]+q[46]-q[4]*q[46])*q[78])
        w[17] = q[11]
        w[18] = (q[11]+q[60]-q[11]*q[60])
        w[19] = q[12]
        w[20] = (q[9]+q[9]*q[48]-q[9]*q[9]*q[48])
        w[21] = ((q[7]+q[8]*q[9]-q[7]*q[8]*q[9])+q[48]*q[9]-(q[7]+q[8]*q[9]-q[7]*q[8]*q[9])*q[48]*q[9])
        w[22] = q[35]
        w[23] = q[22]
        w[24] = q[23]
        w[25] = (q[24]+q[41]-q[24]*q[41])
        w[26] = ((1-q[29])+(1-q[28])-(1-q[29])*(1-q[28]))
        w[27] = ((((q[10]+q[11]-q[10]*q[11])+q[12]-(q[10]+q[11]-q[10]*q[11])*q[12])+q[13]-((q[10]+q[11]-q[10]*q[11])+q[12]-(q[10]+q[11]-q[10]*q[11])*q[12])*q[13])+q[42]*q[48]-(((q[10]+q[11]-q[10]*q[11])+q[12]-(q[10]+q[11]-q[10]*q[11])*q[12])+q[13]-((q[10]+q[11]-q[10]*q[11])+q[12]-(q[10]+q[11]-q[10]*q[11])*q[12])*q[13])*q[42]*q[48])
        w[28] = q[78]*(1-q[0])
        w[29] = q[78]*(1-q[0])
        w[30] = p[1]
        w[31] = p[2]
        w[32] = p[3]
        w[33] = p[3]
        w[34] = p[4]
        w[35] = ((((q[30]+q[3]-q[30]*q[3])+q[32]-(q[30]+q[3]-q[30]*q[3])*q[32])+q[33]-((q[30]+q[3]-q[30]*q[3])+q[32]-(q[30]+q[3]-q[30]*q[3])*q[32])*q[33])+q[34]-(((q[30]+q[3]-q[30]*q[3])+q[32]-(q[30]+q[3]-q[30]*q[3])*q[32])+q[33]-((q[30]+q[3]-q[30]*q[3])+q[32]-(q[30]+q[3]-q[30]*q[3])*q[32])*q[33])*q[34])
        w[36] = q[35]
        w[37] = q[36]
        w[38] = q[37]
        w[39] = (q[38]+q[50]-q[38]*q[50])
        w[40] = q[39]
        w[41] = q[40]
        w[42] = q[25]*(1-q[26])
        w[43] = q[39]
        w[44] = (q[43]+q[78]-q[43]*q[78])
        w[45] = q[44]
        w[46] = q[49]
        w[47] = q[16]
        w[48] = ((q[45]+q[47]-q[45]*q[47])+q[25]*(1-q[26])-(q[45]+q[47]-q[45]*q[47])*q[25]*(1-q[26]))
        w[49] = q[39]
        w[50] = (q[52]+q[97]-q[52]*q[97])
        w[51] = q[3]
        w[52] = (q[31]+q[51]-q[31]*q[51])
        w[53] = ((((q[52]+q[97]*q[96]-q[52]*q[97]*q[96])+q[37]*q[32]-(q[52]+q[97]*q[96]-q[52]*q[97]*q[96])*q[37]*q[32])+q[37]*q[33]-((q[52]+q[97]*q[96]-q[52]*q[97]*q[96])+q[37]*q[32]-(q[52]+q[97]*q[96]-q[52]*q[97]*q[96])*q[37]*q[32])*q[37]*q[33])+q[37]*q[34]-(((q[52]+q[97]*q[96]-q[52]*q[97]*q[96])+q[37]*q[32]-(q[52]+q[97]*q[96]-q[52]*q[97]*q[96])*q[37]*q[32])+q[37]*q[33]-((q[52]+q[97]*q[96]-q[52]*q[97]*q[96])+q[37]*q[32]-(q[52]+q[97]*q[96]-q[52]*q[97]*q[96])*q[37]*q[32])*q[37]*q[33])*q[37]*q[34])
        w[54] = q[53]*q[52]
        w[55] = q[53]*q[37]
        w[56] = (q[55]+q[99]-q[55]*q[99])
        w[57] = (q[9]+q[48]-q[9]*q[48])
        w[58] = q[9]*q[48]
        w[59] = q[9]*q[48]
        w[60] = (q[15]+q[56]-q[15]*q[56])
        w[61] = (((q[73]*q[72]+q[70]-q[73]*q[72]*q[70])+q[41]*q[30]-(q[73]*q[72]+q[70]-q[73]*q[72]*q[70])*q[41]*q[30])+q[41]*q[34]-((q[73]*q[72]+q[70]-q[73]*q[72]*q[70])+q[41]*q[30]-(q[73]*q[72]+q[70]-q[73]*q[72]*q[70])*q[41]*q[30])*q[41]*q[34])
        w[62] = q[61]
        w[63] = (q[62]+q[51]-q[62]*q[51])
        w[64] = q[63]
        w[65] = q[64]
        w[66] = q[4]+q[3]-q[4]*q[3]
        w[67] = (((q[66]+q[74]-q[66]*q[74])+q[77]-(q[66]+q[74]-q[66]*q[74])*q[77])+q[83]*q[84]-((q[66]+q[74]-q[66]*q[74])+q[77]-(q[66]+q[74]-q[66]*q[74])*q[77])*q[83]*q[84])
        w[68] = q[67]
        w[69] = q[67]
        w[70] = q[69]
        w[71] = q[69]
        w[72] = q[70]
        w[73] = q[71]
        w[74] = q[79]
        w[75] = q[85]
        w[76] = q[85]
        w[77] = (q[76]+q[75]-q[76]*q[75])
        w[78] = (q[67]+q[35]-q[67]*q[35])
        w[79] = q[67]
        w[80] = q[78]
        w[81] = q[79]
        w[82] = p[18]
        w[83] = q[82]
        w[84] = q[82]
        w[85] = p[20]
        w[86] = (q[80]+q[9]-q[80]*q[9])
        w[87] = q[86]*(q[80]+q[9]-q[80]*q[9])*(1-q[90])
        w[88] = (q[86]*q[90]+q[86]*(p[12]+q[135]-p[12]*q[135])-q[86]*q[90]*q[86]*(p[12]+q[135]-p[12]*q[135]))
        w[89] = ((q[80]*q[93]+(p[12]+q[135]-p[12]*q[135])-q[80]*q[93]*(p[12]+q[135]-p[12]*q[135]))+((p[15]+q[137]-p[15]*q[137])+q[27]-(p[15]+q[137]-p[15]*q[137])*q[27])-(q[80]*q[93]+(p[12]+q[135]-p[12]*q[135])-q[80]*q[93]*(p[12]+q[135]-p[12]*q[135]))*((p[15]+q[137]-p[15]*q[137])+q[27]-(p[15]+q[137]-p[15]*q[137])*q[27]))
        w[90] = (q[89]+(q[74]+q[80]-q[74]*q[80])*q[93]-q[89]*(q[74]+q[80]-q[74]*q[80])*q[93])*(1-q[87])
        w[91] = (q[87]+q[94]-q[87]*q[94])*p[25]
        w[92] = q[90]*p[32]
        w[93] = q[91]*(1-q[92])
        w[94] = (1-p[35])*q[80]
        w[95] = (p[27]+p[28]-p[27]*p[28])
        w[96] = q[95]
        w[97] = (q[96]+q[100]-q[96]*q[100])
        w[98] = q[50]*q[97]
        w[99] = q[53]*q[97]
        w[100] = (p[29]+q[102]-p[29]*q[102])
        w[101] = q[97]
        w[102] = q[9]
        w[103] = ((q[64]+p[7]-q[64]*p[7]) / (1 + exp((t - p[42]))))*(1-q[116])
        w[104] = ((q[65]+p[8]-q[65]*p[8]) / (1 + exp((t - p[41]))))*(1-q[116])
        w[105] = q[121]*(1-p[39])
        w[106] = q[107]*(1-q[116])
        w[107] = q[112]*q[105]*(1-q[118])
        w[108] = q[107]
        w[109] = (q[106]+q[114]-q[106]*q[114])
        w[110] = q[103]*q[122]*(1-q[116])*(1-p[39])
        w[111] = q[106]*(1-q[116])
        w[112] = q[130]*(1-p[40])
        w[113] = q[131]*(1-p[39])
        w[114] = (q[104]*(1-q[116])+q[123]-q[104]*(1-q[116])*q[123])
        w[115] = q[107]*q[110]
        w[116] = (q[115]+q[136]*q[138]-q[115]*q[136]*q[138])
        w[117] = q[114]
        w[118] = q[112]*(1-q[114])
        w[119] = q[125]*q[127]
        w[120] = q[104]
        w[121] = (q[124]*q[120]*q[119]+q[106]*q[119]-q[124]*q[120]*q[119]*q[106]*q[119])
        w[122] = q[103]*(1-q[116])
        w[123] = ((q[106]+q[104]-q[106]*q[104])+q[128]-(q[106]+q[104]-q[106]*q[104])*q[128])
        w[124] = q[110]
        w[125] = (q[110]+q[106]-q[110]*q[106])
        w[126] = (q[110]+q[106]-q[110]*q[106])
        w[127] = (q[126]+q[125]-q[126]*q[125])
        w[128] = q[127]
        w[129] = q[127]*q[125]
        w[130] = q[129]
        w[131] = q[119]
        w[132] = (p[24]+q[59]-p[24]*q[59])*(p[23]+q[58]-p[23]*q[58])*(p[17]+q[21]-p[17]*q[21])*q[108]*(p[5]+p[6]-p[5]*p[6])*q[145]*q[113]*q[112]*q[105]*p[34]*q[151]*(1-q[135])*(1-q[137])*(1-q[134])
        w[133] = q[132]*q[105]*q[112]*(1-q[134])
        w[134] = (p[24]+q[59]-p[24]*q[59])*(p[12]+p[13]-p[12]*p[13])*q[108]*q[146]*q[111]*q[112]*(1-q[132])*(1-q[138])*(1-q[133])*(1-q[142])
        w[135] = q[134]*(1-q[132])*(1-q[133])
        w[136] = ((p[14]*((p[16]+p[15]-p[16]*p[15])+q[27]-(p[16]+p[15]-p[16]*p[15])*q[27])*q[112]*q[111]*q[105]*q[108]+p[14]*((p[16]+p[15]-p[16]*p[15])+q[27]-(p[16]+p[15]-p[16]*p[15])*q[27])*q[137]*q[115]-p[14]*((p[16]+p[15]-p[16]*p[15])+q[27]-(p[16]+p[15]-p[16]*p[15])*q[27])*q[112]*q[111]*q[105]*q[108]*p[14]*((p[16]+p[15]-p[16]*p[15])+q[27]-(p[16]+p[15]-p[16]*p[15])*q[27])*q[137]*q[115])+p[14]*q[138]-(p[14]*((p[16]+p[15]-p[16]*p[15])+q[27]-(p[16]+p[15]-p[16]*p[15])*q[27])*q[112]*q[111]*q[105]*q[108]+p[14]*((p[16]+p[15]-p[16]*p[15])+q[27]-(p[16]+p[15]-p[16]*p[15])*q[27])*q[137]*q[115]-p[14]*((p[16]+p[15]-p[16]*p[15])+q[27]-(p[16]+p[15]-p[16]*p[15])*q[27])*q[112]*q[111]*q[105]*q[108]*p[14]*((p[16]+p[15]-p[16]*p[15])+q[27]-(p[16]+p[15]-p[16]*p[15])*q[27])*q[137]*q[115])*p[14]*q[138])*(1-q[133])*(1-q[154])*(1-(p[21]+q[57]-p[21]*q[57]))
        w[137] = p[14]*q[136]
        w[138] = q[136]
        w[139] = (((p[21]+q[57]-p[21]*q[57])*p[22]*p[14]*q[105]*q[145]*p[34]+p[22]*p[14]*q[105]*q[145]*q[151]-(p[21]+q[57]-p[21]*q[57])*p[22]*p[14]*q[105]*q[145]*p[34]*p[22]*p[14]*q[105]*q[145]*q[151])+q[154]-((p[21]+q[57]-p[21]*q[57])*p[22]*p[14]*q[105]*q[145]*p[34]+p[22]*p[14]*q[105]*q[145]*q[151]-(p[21]+q[57]-p[21]*q[57])*p[22]*p[14]*q[105]*q[145]*p[34]*p[22]*p[14]*q[105]*q[145]*q[151])*q[154])*(1-q[132])*(1-q[136])*(1-q[134])
        w[140] = (p[22]*q[139]+(p[21]+q[57]-p[21]*q[57])*q[142]-p[22]*q[139]*(p[21]+q[57]-p[21]*q[57])*q[142])*(1-q[133])*(1-q[135])*(1-q[137])
        w[141] = q[139]
        w[142] = (p[21]+q[57]-p[21]*q[57])*p[22]*q[105]*q[145]*(1-q[139])*(1-q[132])*(1-q[134])
        w[143] = q[142]
        w[144] = q[142]
        w[145] = (q[109]*q[114]+q[109]*q[151]-q[109]*q[114]*q[109]*q[151])*(1-q[148])*(1-p[38])
        w[146] = (q[109]*q[148]+q[109]*(p[12]+p[13]-p[12]*p[13])-q[109]*q[148]*q[109]*(p[12]+p[13]-p[12]*p[13]))
        w[147] = q[114]*q[153]
        w[148] = (((((q[147]*(1-q[145])+q[130]*q[153]*(1-q[145])-q[147]*(1-q[145])*q[130]*q[153]*(1-q[145]))+q[114]*q[153]*(1-q[145])-(q[147]*(1-q[145])+q[130]*q[153]*(1-q[145])-q[147]*(1-q[145])*q[130]*q[153]*(1-q[145]))*q[114]*q[153]*(1-q[145]))+q[136]-((q[147]*(1-q[145])+q[130]*q[153]*(1-q[145])-q[147]*(1-q[145])*q[130]*q[153]*(1-q[145]))+q[114]*q[153]*(1-q[145])-(q[147]*(1-q[145])+q[130]*q[153]*(1-q[145])-q[147]*(1-q[145])*q[130]*q[153]*(1-q[145]))*q[114]*q[153]*(1-q[145]))*q[136])+q[142]-(((q[147]*(1-q[145])+q[130]*q[153]*(1-q[145])-q[147]*(1-q[145])*q[130]*q[153]*(1-q[145]))+q[114]*q[153]*(1-q[145])-(q[147]*(1-q[145])+q[130]*q[153]*(1-q[145])-q[147]*(1-q[145])*q[130]*q[153]*(1-q[145]))*q[114]*q[153]*(1-q[145]))+q[136]-((q[147]*(1-q[145])+q[130]*q[153]*(1-q[145])-q[147]*(1-q[145])*q[130]*q[153]*(1-q[145]))+q[114]*q[153]*(1-q[145])-(q[147]*(1-q[145])+q[130]*q[153]*(1-q[145])-q[147]*(1-q[145])*q[130]*q[153]*(1-q[145]))*q[114]*q[153]*(1-q[145]))*q[136])*q[142])+p[37]-((((q[147]*(1-q[145])+q[130]*q[153]*(1-q[145])-q[147]*(1-q[145])*q[130]*q[153]*(1-q[145]))+q[114]*q[153]*(1-q[145])-(q[147]*(1-q[145])+q[130]*q[153]*(1-q[145])-q[147]*(1-q[145])*q[130]*q[153]*(1-q[145]))*q[114]*q[153]*(1-q[145]))+q[136]-((q[147]*(1-q[145])+q[130]*q[153]*(1-q[145])-q[147]*(1-q[145])*q[130]*q[153]*(1-q[145]))+q[114]*q[153]*(1-q[145])-(q[147]*(1-q[145])+q[130]*q[153]*(1-q[145])-q[147]*(1-q[145])*q[130]*q[153]*(1-q[145]))*q[114]*q[153]*(1-q[145]))*q[136])+q[142]-(((q[147]*(1-q[145])+q[130]*q[153]*(1-q[145])-q[147]*(1-q[145])*q[130]*q[153]*(1-q[145]))+q[114]*q[153]*(1-q[145])-(q[147]*(1-q[145])+q[130]*q[153]*(1-q[145])-q[147]*(1-q[145])*q[130]*q[153]*(1-q[145]))*q[114]*q[153]*(1-q[145]))+q[136]-((q[147]*(1-q[145])+q[130]*q[153]*(1-q[145])-q[147]*(1-q[145])*q[130]*q[153]*(1-q[145]))+q[114]*q[153]*(1-q[145])-(q[147]*(1-q[145])+q[130]*q[153]*(1-q[145])-q[147]*(1-q[145])*q[130]*q[153]*(1-q[145]))*q[114]*q[153]*(1-q[145]))*q[136])*q[142])*p[37])
        w[149] = (q[145]*p[26]+q[154]*p[26]-q[145]*p[26]*q[154]*p[26])*(1-q[153])*(1-q[142])
        w[150] = p[31]
        w[151] = q[150]
        w[152] = q[148]*p[33]
        w[153] = q[149]*(1-q[152])
        w[154] = (1-p[36])*q[114]
        return w

    def create_ode_system(self, af=None, decay_rates=None, threshold_values=None):
        def act_f(a_function=None, x=None, b=None, u=None):
            if a_function is None:
                a_function = 'sigmoid'
            if a_function == 'sigmoid':
                return 1 / (1 + np.exp(-b*(x-u)))
            elif a_function == 'tanh':
                return (1+np.tanh(-b*(x-u)))/2
            elif a_function == 'hill':
                return hill_activation(x, n=10, Kd=0.5, u=0.5)
        def hill_activation(x, n=10, Kd=0.5, u=0.5):
            x = np.asarray(x)
            return x**n / (Kd**n + x**n)
        def system_ode(t, q, *p):
            w = self.interaction_terms(t, q, p, np.zeros(len(q)))
            dqdt = act_f(af, x=w, b=p[0], u=threshold_values) - decay_rates*q
            return dqdt

        return system_ode

    def solve(self, t_span=(0, 55), t_eval=None, grinitial_conditions=None, grmethod=None, grout_of_the_network_arguments_eval=None, af=None, decay_rates_list=None, grthreshold_values=None):
        if grinitial_conditions is None:
            grinitial_conditions = self.initial_conditions
        if grmethod is None:
            grmethod = 'LSODA'
        if grout_of_the_network_arguments_eval is None:
            grout_of_the_network_arguments_eval = self.initial_out
        if decay_rates_list is None:
            decay_rates_list = np.full(self.n, 1)
        if af is None:
            af = 'sigmoid'
        if grthreshold_values is None:
            grthreshold_values = np.full(self.n, 0.5)

        ode_system = self.create_ode_system(af, decay_rates_list, grthreshold_values)
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 500)

        if self.out_of_network_arguments is None:
            sol = solve_ivp(
                ode_system,
                t_span,
                grinitial_conditions,
                method=grmethod,
                t_eval=t_eval,
                dense_output=True
            )
        else:
            sol = solve_ivp(
                ode_system,
                t_span,
                grinitial_conditions,
                args=grout_of_the_network_arguments_eval,
                method=grmethod,
                t_eval=t_eval,
                dense_output=True
            )

        results = {
            'time': sol.t,
            'nodes': {}
        }
        for i, node in enumerate(self.array_of_elements):
            results['nodes'][node] = {
                'values': sol.y[i],
                'initial': sol.y[i, 0],
                'final': sol.y[i, -1],
                'max': np.max(sol.y[i]),
                'min': np.min(sol.y[i]),
                'mean': np.mean(sol.y[i])
            }

        return results
    def solve_ito(self,
              t_span=(0, 55),
              t_eval=None,
              grinitial_conditions=None,
              grout_of_the_network_arguments_eval=None,
              af='sigmoid',
              decay_rates_list=None,
              grthreshold_values=None,
              noise_sigmas=None,
              seed=None):
        # --- Defaults ---
        if grinitial_conditions is None:
            grinitial_conditions = self.initial_conditions
        y0 = np.asarray(grinitial_conditions, dtype=float)

        if grout_of_the_network_arguments_eval is None:
            grout_of_the_network_arguments_eval = self.initial_out

        if decay_rates_list is None:
            decay_rates_list = np.full(self.n, 1.0)

        if grthreshold_values is None:
            grthreshold_values = np.full(self.n, 0.5)

        if noise_sigmas is None:
            noise_sigmas = np.zeros(self.n)

        sigmas = np.asarray(noise_sigmas, dtype=float)
        if sigmas.shape[0] != self.n:
            raise ValueError(f"noise_sigmas debe tener {self.n} elementos")
        ode_system = self.create_ode_system(af, decay_rates_list, grthreshold_values)
        p = grout_of_the_network_arguments_eval

        def drift(y, t):
            return ode_system(t, y, *p)
        def diffusion(y, t):
            return np.diag(sigmas)
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 500)
        if seed is not None:
            np.random.seed(seed)
        sol = sdeint.itoint(drift, diffusion, y0, t_eval)
        results = {
            'time': t_eval,
            'nodes': {},
            'stochastic': True,
        }
        for i, node in enumerate(self.array_of_elements):
            results['nodes'][node] = {
                'values': sol[:, i],
                'initial': sol[0, i],
                'final': sol[-1, i],
                'max': float(np.max(sol[:, i])),
                'min': float(np.min(sol[:, i])),
                'mean': float(np.mean(sol[:, i])),
            }
        return results
