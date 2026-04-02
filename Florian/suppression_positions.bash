cp resultat.csv x.csv
grep -v 'DeepAutoFocus/Z-stacks/Zeiss_30zStack0-2-30_4/Pos25/' x.csv > y.csv; mv y.csv x.csv
grep -v 'DeepAutoFocus/Z-stacks/Zeiss_30zStack0-2-30_4/Pos16/' x.csv > y.csv; mv y.csv x.csv
grep -v 'DeepAutoFocus/Z-stacks/Nikon/020525_3119s_GT_empty/' x.csv > y.csv; mv y.csv x.csv
grep -v 'DeepAutoFocus/Z-stacks/Nikon/20250610_Nikon_zStacks_W52_YAK1-09/_2/8-Pos002_005/' x.csv > y.csv; mv y.csv x.csv
grep -v 'DeepAutoFocus/Z-stacks/Nikon/20250610_Nikon_zStacks_W52_YAK1-09/_2/8-Pos003_005/' x.csv > y.csv; mv y.csv x.csv
grep -v 'DeepAutoFocus/Z-stacks/Nikon/20250610_Nikon_zStacks_W52_YAK1-09/_2/8-Pos004_005/' x.csv > y.csv; mv y.csv x.csv
grep -v 'DeepAutoFocus/Z-stacks/Nikon/20250610_Nikon_zStacks_W52_YAK1-09/_3/9-Pos000_000/' x.csv > y.csv; mv y.csv x.csv
grep -v 'DeepAutoFocus/Z-stacks/Nikon/20250610_Nikon_zStacks_W52_YAK1-09/_5/3-Pos000_000/' x.csv > y.csv; mv y.csv x.csv
mv x.csv resultat_propre.csv
