from plotting import PlotConfig
SUBNETWORKS = {
    'Macrophage inputs': PlotConfig(
        variables=[],
        labels=[],
        colors=[],
        styles=[],
        legfontsize=12,
        legncol=1,
        title='Macrophage environment'
    ),
    'Macrophage Metabolism': PlotConfig(
        variables=['MGlycolysis', 'MOXPHOS', 'MAMPATPratio', 'MAMPK'],
        labels=['Glycolysis', 'OXPHOS', 'AMP/ATP', 'AMPK'],
        colors=['lawngreen', 'blueviolet', 'tab:cyan', 'tab:red'],
        styles=['-', ':', '--', '-.'],
        legfontsize=12,
        legncol=1,
        title='Metabolism'
    ),
    'Phagocytosis': PlotConfig(
        variables=['PHAGOSOME','PHAGOCYTOSIS', 'PROCESSING', 'MHC2'],
        labels=['Endocytosis','Phagocytic program', 'Antigen Processing', 'MHC II-CD80/86'],
        colors=['#FFD700', 'brown', '#1E90FF', '#FF6347'],
        styles=['-', '-.', '-', '-'],
        legfontsize=12,
        legncol=1,
        title='Phagocytosis'
    ),
    'Macrophage transcription factors': PlotConfig(
        variables=['MNFKB', 'MAP1', 'IRF3', 'IRF4', 'IRF7'],
        labels=['NF-κB', 'AP-1', 'IRF3', 'IRF4', 'IRF7'],
        colors=['#6A5ACD', '#FF69B4', '#00FA9A', '#FF4500', '#9932CC'],
        styles=['-.', '-', '-', '-', '-'],
        legfontsize=12,
        legncol=2,
        title='Transcription factors'
    ),
    'Secreted cytokines': PlotConfig(
        variables=['IL6OUT', 'IL12OUT', 'IL18OUT', 'IL33OUT', 'IL10OUT','IFNA','IL1BOUT', 'TNFAOUT'],
        labels=['IL-6', 'IL-12', 'IL-18', 'IL-33', 'IL-10','IFN-α/β','IL-1β','TNF-α'],
        colors=['brown','tab:brown', 'red', 'blue', 'orange', 'blueviolet', '#3CB371', 'black'],
        styles=['-.','-.', '-.', '-', '-.', ':', ':', ':'],
        legfontsize=12,
        legncol=2,
        title='Fully secreted cytokines'
    ),
    'M1': PlotConfig(
        variables=['IFNA', 'IL1BOUT', 'IL6OUT', 'IL12OUT','IL18OUT', 'IL33OUT',  'TNFAOUT'],
        labels=['IFN-α/β', 'IL-1β', 'IL6', 'IL12','IL18', 'IL33',  'TNF-α'],
        colors=['blueviolet', '#3CB371','brown','tab:brown', 'red', 'blue', 'black'],
        styles=[':', ':', '-.', '-.', '-.', '-', ':'],
        legfontsize=11,
        legncol=2,
        title='M1'
    ),
    'M2a': PlotConfig(
        variables=['PPARG', 'IL10OUT', 'JMJD3', 'STAT6'],
        labels=['PPAR-γ', 'IL10', 'JMJD3', 'STAT6'],
        colors=['#DAA520','blueviolet', '#20B2AA', '#CD5C5C'],
        styles=['-', ':', '-', '-'],
        legfontsize=12,
        legncol=2,
        title='M2a'
    ),
    'M2b': PlotConfig(
        variables=['ERK', 'IL10OUT' ],
        labels=['ERK', 'IL10'],
        colors=['#B8860B', 'blueviolet'],
        styles=['-', ':'],
        legfontsize=12,
        legncol=1,
        title='M2b'
    ),
    'M2c': PlotConfig(
        variables=['STAT3', 'IL10OUT'],
        labels=['STAT3', 'IL10'],
        colors=['#DB7093','blueviolet'],
        styles=['-', ':'],
        legfontsize=12,
        legncol=1,
        title='M2c'
    ),
    'Lymphocyte phenotype': PlotConfig(
        variables=['TBET', 'GATA3', 'RORGT', 'FOXP3', 'BCL6'],
        labels=['Th1', 'Th2', 'Th17', 'Treg', 'Tfh'],
        colors=['#228B22', '#4B0082', '#008080', '#B22222', '#A0522D'],
        styles=['-', '-', '-', '-', '-'],
        legfontsize=12,
        legncol=1,
        title='Lymphocyte phenotype'
    ),
    'Lymphocyte inputs': PlotConfig(
        variables=[],
        labels=[],
        colors=[],
        styles=[],
        legfontsize=12,
        legncol=1,
        title='T cell environment'
    ),
    'Antigen presentation': PlotConfig(
        variables=['TCR', 'CD28', 'CTLA4', 'NDRG1'],
        labels=['TCR', 'CD28', 'CTLA4', 'NDRG1'],
        colors=['#7B68EE', '#48D1CC', '#FFA07A', '#F08080'],
        styles=['-', '-', '-', '-'],
        legfontsize=12,
        legncol=2,
        title='Antigen presentation'
    ),
    'T Metabolism': PlotConfig(
        variables=['Glycolysis', 'OXPHOS', 'AMPATPratio', 'AMPK'],
        labels=['Glycolysis', 'OXPHOS', 'AMP/ATP', 'AMPK'],
        colors=['lawngreen', 'blueviolet', 'tab:cyan', 'tab:red'],
        styles=['-', ':', '--', '-.'],
        legfontsize=12,
        legncol=1,
        title='Metabolism'
    ),
    'T Transcription factors': PlotConfig(
        variables=['AP1', 'NFAT', 'NFKB'],
        labels=['AP1', 'NFAT', 'NFKB'],
        colors=['#FF69B4', '#FFB6C1', 'black'],
        styles=['-', '-', '-.'],
        legfontsize=12,
        legncol=2,
        title='Transcription factors'
    ),
    'Activation markers': PlotConfig(
        variables=['IL2G', 'MTORC1', 'MTORC2'],
        labels=['IL2G', 'MTORC1', 'MTORC2'],
        colors=['#FA8072', '#F4A460', '#C71585'],
        styles=['-', '-', '-'],
        legfontsize=12,
        legncol=2,
        title='Activation markers'
    ),
    'Th1': PlotConfig(
        variables=['TBET', 'IFNG'],
        labels=['TBET', 'IFN-γ'],
        colors=['#228B22', '#808000'],
        styles=['-', '-'],
        legfontsize=12,
        legncol=1,
        title='Th1'
    ),
    'Th2': PlotConfig(
        variables=['GATA3', 'IL4'],
        labels=['GATA3', 'IL4'],
        colors=['#4B0082', '#556B2F'],
        styles=['-', ':'],
        legfontsize=12,
        legncol=1,
        title='Th2'
    ),
    'Th17': PlotConfig(
        variables=['IL17', 'IL21', 'RORGT'],
        labels=['IL17', 'II21', 'RORGT'],
        colors=['#FF8C00', '#E9967A', '#008080'],
        styles=[':', ':', '-'],
        legfontsize=12,
        legncol=2,
        title='Th17'
    ),
    'Treg': PlotConfig(
        variables=['FOXP3', 'TGFB', 'CTLA4DIM','IL10OUT'],
        labels=['FOXP3', 'TGFB', 'CTLA4D','IL10'],
        colors=['#B22222', '#DAA520', '#BDB76B', '#9370DB'],
        styles=['-', '-', '-', ':'],
        legfontsize=12,
        legncol=2,
        title='Treg'
    ),
    'Tfh': PlotConfig(
        variables=['BCL6', 'IL21', 'CD40L', 'IL9'],
        labels=['BCL6', 'IL21', 'CD40L', 'IL9'],
        colors=['#A0522D', '#E9967A', '#66CDAA', '#8FBC8F'],
        styles=['-', ':', '-', ':'],
        legfontsize=12,
        legncol=2,
        title='Tfh'
    )
}
