clear version computer ver
workDir = "D:\driveMatlab\Academics\arpa6sirali";
assert(isfolder(workDir), ...
"Çalışma klasörü bulunamadı: %s", workDir);
cd(workDir);
rng(42,"twister");
T = readtable( ...
'arpa_veriseti.xlsx', ...
'VariableNamingRule','preserve');
disp('Tablo boyutu:');
disp(size(T));
disp('Tablo sütun isimleri:');
disp(T.Properties.VariableNames);
head(T)
matlabVersion = string(feval('version'));
matlabRelease = string(feval('version','-release'));
matlabPlatform = string(feval('computer'));
matlabInfo = table( ...
    matlabVersion, ...
    matlabRelease, ...
    matlabPlatform, ...
'VariableNames', ...
    {'MATLAB_Version','Release','Platform'});
matlabInfo
statsToolboxAvailable = ~isempty(ver('stats'));
assert(statsToolboxAvailable, ...
    ['Statistics and Machine Learning Toolbox gereklidir ' ...
'(cvpartition, fitrsvm, fitrensemble, fitrgp, pca, kmeans, friedman, signrank).']);
pythonStatus = checkPythonBridge();
pythonStatus
cfg = arpaConfig(workDir);
cfg.FAST_MODE = false;
cfg.RESUME_FROM_CHECKPOINT = true;
cfg.RUN_ABLATION = false;
cfg.RUN_FINAL_MODELS = true;
cfg.LEAKAGE_POLICY = "legacy";
cfg.FINAL_FEATURE_SET_KEY = "A2_SAFE_ENGINEERED";
cfg.MANUAL_GROUP_COLUMN = "";
cfg = applyModeSettings(cfg);
cfg
if ~exist("cfg","var"), cfg = arpaConfig(); end
[T,y] = loadArpaData(cfg);
fprintf('[Bilgi] Veri dosyası   : %s\n',cfg.DATA_PATH);
fprintf('[Bilgi] Çıktı klasörü : %s\n',cfg.OUTPUT_DIR);
fprintf('[Bilgi] Gözlem sayısı : %d\n',height(T));
fprintf('[Bilgi] Sütun sayısı  : %d\n',width(T));
head(T)
if ~exist("cfg","var"), cfg = arpaConfig(); end
if ~exist("T","var") || ~exist("y","var"), [T,y] = loadArpaData(cfg); end
[schemaTable,categoricalFeatures] = inspectSchema(T,cfg);
schemaTable
writetable(schemaTable,fullfile(cfg.OUTPUT_DIR,"veri_sozlugu_kategorik_denetim.csv"));
f = figure('Name','Eksik Değerler');
bar(categorical(schemaTable.Sutun,schemaTable.Sutun),schemaTable.Eksik);
ylabel('Eksik gözlem sayısı');
title('Model girdilerinde eksik değerler');
xtickangle(45); grid on;
exportgraphics(f,fullfile(cfg.OUTPUT_DIR,"veri_eksik_degerler.png"),'Resolution',200);
if ~exist("cfg","var"), cfg = arpaConfig(); end
if ~exist("T","var") || ~exist("y","var"), [T,y] = loadArpaData(cfg); end
[leakageAudit,rawExclusions] = auditHIFormulas(T,y,cfg);
leakageAudit
writetable(leakageAudit,fullfile(cfg.OUTPUT_DIR,"HI_matematiksel_iliskiler_denetimi.csv"));
fprintf('[Bilgi] Sızıntı politikası: %s\n',cfg.LEAKAGE_POLICY);
fprintf('[Bilgi] Dışlanan ham özellikler: %s\n',strjoin(rawExclusions,', '));
if ~isempty(leakageAudit)
    f = figure('Name','HI Matematiksel Sızıntı Denetimi');
    tiledlayout(1,2);
    nexttile; bar(categorical(leakageAudit.Aday_formul,leakageAudit.Aday_formul),leakageAudit.Pearson_r); ylim([-1 1]); ylabel('Pearson r'); title('HI ile korelasyon'); xtickangle(30); grid on;
    nexttile; bar(categorical(leakageAudit.Aday_formul,leakageAudit.Aday_formul),leakageAudit.Kalibre_R2); ylim([0 1]); ylabel('Kalibre R^2'); title('Basit kalibratör açıklama gücü'); xtickangle(30); grid on;
    exportgraphics(f,fullfile(cfg.OUTPUT_DIR,"HI_matematiksel_sizinti_denetimi.png"),'Resolution',200);
end
if ~exist("cfg","var"), cfg = arpaConfig(); end
if ~exist("T","var") || ~exist("y","var"), [T,y] = loadArpaData(cfg); end
[groupColumn,groups] = inferGroupColumn(T,cfg);
if strlength(groupColumn)==0
    fprintf('[Bilgi] Grup sütunu kullanılmadı. %d×%d Repeated K-Fold uygulanacak.\n',cfg.OUTER_REPEATS,cfg.OUTER_SPLITS);
    fprintf('[Uyarı] Aynı genotipe ait tekrarlı satırlar varsa cfg.MANUAL_GROUP_COLUMN değerini tanımlayın.\n');
else
    fprintf('[Bilgi] Grup-korumalı CV: %s (%d grup).\n',groupColumn,numel(unique(groups)));
end
if ~exist("cfg","var"), cfg = arpaConfig(); end
if ~exist("T","var") || ~exist("y","var"), [T,y] = loadArpaData(cfg); end
if ~exist("rawExclusions","var"), [leakageAudit,rawExclusions] = auditHIFormulas(T,y,cfg); end
featureSpecs = buildFeatureSpecs(cfg,rawExclusions);
featureSpecTable = featureSpecsToTable(featureSpecs);
featureSpecTable
writetable(featureSpecTable,fullfile(cfg.OUTPUT_DIR,"feature_sets.csv"));
pcaKmeansProtocol = table(4,7,20,true, ...
'VariableNames',{'PCA_Bilesen','KMeans_Kume','KMeans_Replicate','Orijinal_Ozellikleri_Koru'});
pcaKmeansProtocol
kanProtocol = table(5,3,"[-3, 3]",128,140,20,0.15,5, ...
'VariableNames',{'Varsayilan_Grid','Varsayilan_Spline_Order','Grid_Araligi','Batch','MaxEpoch_Full','Patience_Full','ValidationFraction','GradientClip'});
kanProtocol
if ~exist("cfg","var"), cfg = arpaConfig(); end
modelMap = modelImplementationTable();
modelMap
writetable(modelMap,fullfile(cfg.OUTPUT_DIR,"model_matlab_eslestirmesi.csv"));
if ~exist("cfg","var"), cfg = arpaConfig(); end
hyperparameterTable = hyperparameterSpaceTable(cfg);
hyperparameterTable
writetable(hyperparameterTable,fullfile(cfg.OUTPUT_DIR,"hiperparametre_uzaylari.csv"));
if ~exist("cfg","var"), cfg = arpaConfig(); end
if ~exist("T","var") || ~exist("y","var"), [T,y] = loadArpaData(cfg); end
if ~exist("groupColumn","var") || ~exist("groups","var"), [groupColumn,groups] = inferGroupColumn(T,cfg); end
outerSplits = makeOuterSplits(height(T),groups,cfg);
outerFoldAudit = summarizeOuterSplits(outerSplits,groups);
outerFoldAudit
writetable(outerFoldAudit,fullfile(cfg.OUTPUT_DIR,"outer_fold_audit.csv"));
disp('optimizeOnOuterTrain(modelName, featureSpec, Ttrain, ytrain, groupsTrain, foldSeed, nTrials, cfg)');
optimizeOnOuterTrain(modelName, featureSpec, Ttrain, ytrain, groupsTrain, foldSeed, nTrials, cfg)
disp('nestedCVEvaluate(modelName, featureKey, featureSpec, T, y, groups, outerSplits, nTrials, stage, cfg)');
nestedCVEvaluate(modelName, featureKey, featureSpec, T, y, groups, outerSplits, nTrials, stage, cfg)
if ~exist("cfg","var"), cfg = arpaConfig(); end
if ~exist("rawExclusions","var")
if ~exist("T","var") || ~exist("y","var"), [T,y] = loadArpaData(cfg); end
    [leakageAudit,rawExclusions] = auditHIFormulas(T,y,cfg);
end
if ~exist("groupColumn","var") || ~exist("groups","var"), [groupColumn,groups] = inferGroupColumn(T,cfg); end
[resultTag,configFingerprint] = makeResultTag(cfg,rawExclusions,groupColumn);
fprintf('[Bilgi] Çalışma yapı etiketi: %s\n',resultTag);
fprintf('[Bilgi] Fingerprint: %s\n',configFingerprint);
if ~exist("cfg","var"), cfg = arpaConfig(); end
if ~exist("T","var") || ~exist("y","var"), [T,y] = loadArpaData(cfg); end
if ~exist("rawExclusions","var"), [leakageAudit,rawExclusions] = auditHIFormulas(T,y,cfg); end
if ~exist("featureSpecs","var"), featureSpecs = buildFeatureSpecs(cfg,rawExclusions); end
if ~exist("groupColumn","var") || ~exist("groups","var"), [groupColumn,groups] = inferGroupColumn(T,cfg); end
if ~exist("outerSplits","var"), outerSplits = makeOuterSplits(height(T),groups,cfg); end
if ~exist("resultTag","var"), [resultTag,configFingerprint] = makeResultTag(cfg,rawExclusions,groupColumn); end
ablationFolds = table();
ablationPredictions = table();
if cfg.RUN_ABLATION
for fsKey = string(fieldnames(featureSpecs))'
for modelName = cfg.ABLATION_MODELS
            nTrials = max(3,floor(getTrialCount(cfg,modelName)/2));
            [fr,pr] = runOrLoadNested("ablation",fsKey,modelName,nTrials,featureSpecs,T,y,groups,outerSplits,resultTag,cfg);
            ablationFolds = [ablationFolds; fr];
            ablationPredictions = [ablationPredictions; pr];
end
end
    writetable(ablationFolds,fullfile(cfg.OUTPUT_DIR,"ablation_all_fold_results_"+resultTag+".csv"));
    writetable(ablationPredictions,fullfile(cfg.OUTPUT_DIR,"ablation_all_predictions_"+resultTag+".csv"));
else
    disp('[Bilgi] RUN_ABLATION=false; ablation yeniden çalıştırılmadı.');
end
ablationSummary = summarizeFoldResults(ablationFolds);
ablationSummary
ablationTests = table();
if ~isempty(ablationFolds)
    ablationTests = ablationPairedTests(ablationFolds,cfg);
    ablationTests
    writetable(ablationSummary,fullfile(cfg.OUTPUT_DIR,"ablation_summary_"+resultTag+".csv"));
    writetable(ablationTests,fullfile(cfg.OUTPUT_DIR,"ablation_wilcoxon_holm_"+resultTag+".csv"));
end
if ~isempty(ablationSummary)
    f = plotAblation(ablationSummary,featureSpecs,cfg);
    exportgraphics(f,fullfile(cfg.OUTPUT_DIR,"ablation_R2_"+resultTag+".png"),'Resolution',200);
end
if ~exist("cfg","var"), cfg = arpaConfig(); end
if ~exist("T","var") || ~exist("y","var"), [T,y] = loadArpaData(cfg); end
if ~exist("rawExclusions","var"), [leakageAudit,rawExclusions] = auditHIFormulas(T,y,cfg); end
if ~exist("featureSpecs","var"), featureSpecs = buildFeatureSpecs(cfg,rawExclusions); end
if ~exist("groupColumn","var") || ~exist("groups","var"), [groupColumn,groups] = inferGroupColumn(T,cfg); end
if ~exist("outerSplits","var"), outerSplits = makeOuterSplits(height(T),groups,cfg); end
if ~exist("resultTag","var"), [resultTag,configFingerprint] = makeResultTag(cfg,rawExclusions,groupColumn); end
finalFolds = table();
finalPredictions = table();
fprintf('[Bilgi] Nihai özellik seti: %s\n',cfg.FINAL_FEATURE_SET_KEY);
fprintf('[Bilgi] Model sayısı: %d\n',numel(cfg.CORE_MODELS));
if cfg.RUN_FINAL_MODELS
for modelName = cfg.CORE_MODELS
        nTrials = getTrialCount(cfg,modelName);
        [fr,pr] = runOrLoadNested("final",cfg.FINAL_FEATURE_SET_KEY,modelName,nTrials,featureSpecs,T,y,groups,outerSplits,resultTag,cfg);
        finalFolds = [finalFolds; fr];
        finalPredictions = [finalPredictions; pr];
end
    writetable(finalFolds,fullfile(cfg.OUTPUT_DIR,"final_all_fold_results_"+resultTag+".csv"));
    writetable(finalPredictions,fullfile(cfg.OUTPUT_DIR,"final_all_predictions_"+resultTag+".csv"));
else
    disp('[Bilgi] RUN_FINAL_MODELS=false; nihai modeller çalıştırılmadı.');
end
finalSummary = finalPerformanceSummary(finalFolds,cfg.SEED);
finalSummary
if ~isempty(finalSummary)
    writetable(finalSummary,fullfile(cfg.OUTPUT_DIR,"final_model_summary_"+resultTag+".csv"));
    writetable(finalSummary,fullfile(cfg.OUTPUT_DIR,"MATLAB_sonuclar_"+resultTag+".xlsx"),'Sheet','Final_Summary','WriteMode','overwritesheet');
end
writetable(finalSummary,'finalSummary.xlsx');
friedmanResult = table();
averageRanks = table();
posthocResults = table();
if ~isempty(finalFolds)
    [friedmanResult,averageRanks,posthocResults] = finalStatisticalTests(finalFolds);
    friedmanResult
    averageRanks
    posthocResults
    writetable(friedmanResult,fullfile(cfg.OUTPUT_DIR,"friedman_test_"+resultTag+".csv"));
    writetable(averageRanks,fullfile(cfg.OUTPUT_DIR,"average_model_ranks_"+resultTag+".csv"));
    writetable(posthocResults,fullfile(cfg.OUTPUT_DIR,"wilcoxon_posthoc_holm_"+resultTag+".csv"));
end
if ~isempty(finalFolds) && ~isempty(finalSummary)
    f = plotFinalR2Boxplot(finalFolds,finalSummary,cfg.FINAL_FEATURE_SET_KEY);
    exportgraphics(f,fullfile(cfg.OUTPUT_DIR,"final_model_R2_boxplot_"+resultTag+".png"),'Resolution',200);
end
if ~isempty(finalPredictions) && ~isempty(finalSummary)
    bestModel = string(finalSummary.Model(1));
    f = plotActualPredicted(finalPredictions,bestModel);
    exportgraphics(f,fullfile(cfg.OUTPUT_DIR,"best_model_actual_vs_predicted_"+resultTag+".png"),'Resolution',200);
end
if ~isempty(averageRanks)
    f = figure('Name','Ortalama Model Sıraları');
    bar(categorical(averageRanks.Model,averageRanks.Model),averageRanks.average_rank);
    ylabel('Ortalama sıra (1 = en iyi)'); xlabel('Model'); title('Nested CV — Ortalama Model Sıraları'); xtickangle(35); grid on;
    exportgraphics(f,fullfile(cfg.OUTPUT_DIR,"average_model_ranks_"+resultTag+".png"),'Resolution',200);
end
RUN_SYNTHETIC_APPENDIX = false;
if RUN_SYNTHETIC_APPENDIX
error('Sentetik deney ayrı ve önceden tanımlı bir protokolle yürütülmelidir.');
else
    disp('[Bilgi] Sentetik veri üretimi ana analizde çalıştırılmadı (beklenen davranış).');
end
if ~exist("cfg","var"), cfg = arpaConfig(); end
if ~exist("T","var") || ~exist("y","var"), [T,y] = loadArpaData(cfg); end
if ~exist("rawExclusions","var"), [leakageAudit,rawExclusions] = auditHIFormulas(T,y,cfg); end
if ~exist("groupColumn","var"), [groupColumn,groups] = inferGroupColumn(T,cfg); end
if ~exist("resultTag","var"), [resultTag,configFingerprint] = makeResultTag(cfg,rawExclusions,groupColumn); end
[manifestPath,zipPath] = saveManifestAndArchive(cfg,rawExclusions,groupColumn,resultTag,configFingerprint);
fprintf('[Tamamlandı] Manifest: %s\n',manifestPath);
fprintf('[Tamamlandı] ZIP     : %s\n',zipPath);
outputFile = fullfile(pwd,'MATLAB_arpa_tum_sonuclar.xlsx');
if isfile(outputFile)
    delete(outputFile);
end
writetable(finalSummary,outputFile,'Sheet','FinalSummary');
writetable(finalFolds,outputFile,'Sheet','FinalFolds');
writetable(finalPredictions,outputFile,'Sheet','FinalPredictions');
writetable(averageRanks,outputFile,'Sheet','AverageRanks');
writetable(friedmanResult,outputFile,'Sheet','Friedman');
writetable(posthocResults,outputFile,'Sheet','PostHoc_Wilcoxon');
writetable(leakageAudit,outputFile,'Sheet','LeakageAudit');
writetable(outerFoldAudit,outputFile,'Sheet','OuterFoldAudit');
writetable(featureSpecTable,outputFile,'Sheet','FeatureSpecs');
writetable(hyperparameterTable,outputFile,'Sheet','Hyperparameters');
writetable(modelMap,outputFile,'Sheet','ModelMap');
writetable(schemaTable,outputFile,'Sheet','Schema');
writetable(pcaKmeansProtocol,outputFile,'Sheet','PCA_KMeans');
writetable(kanProtocol,outputFile,'Sheet','KAN_Protocol');
writetable(matlabInfo,outputFile,'Sheet','MATLAB_Info');
writetable(pythonStatus,outputFile,'Sheet','Python_Info');
fprintf('\n============================================\n');
fprintf('Tum MATLAB sonuclari Excel dosyasina kaydedildi:\n');
fprintf('%s\n',outputFile);
fprintf('============================================\n');
save('MATLAB_arpa_workspace.mat','-v7.3')
function cfg = arpaConfig(workDir)
if nargin<1 || strlength(string(workDir))==0
    workDir = "D:\driveMatlab\Academics\arpa6sirali";
end
cfg = struct();
cfg.CODE_VERSION = "v27.0-matlab-r2026a";
cfg.WORK_DIR = string(workDir);
cfg.DATA_PATH = fullfile(cfg.WORK_DIR,"arpa_veriseti.xlsx");
cfg.OUTPUT_DIR_NAME = "tez_analiz_v27_SGW_SPW_dahil_outputs";
cfg.OUTPUT_DIR = fullfile(cfg.WORK_DIR,cfg.OUTPUT_DIR_NAME);
if ~isfolder(cfg.OUTPUT_DIR), mkdir(cfg.OUTPUT_DIR); end
cfg.FAST_MODE = false;
cfg.RESUME_FROM_CHECKPOINT = true;
cfg.RUN_ABLATION = false;
cfg.RUN_FINAL_MODELS = true;
cfg.TARGET = "HI";
cfg.MANUAL_GROUP_COLUMN = "";
cfg.LEAKAGE_POLICY = "legacy";
cfg.FINAL_FEATURE_SET_KEY = "A2_SAFE_ENGINEERED";
cfg.SEED = 42;
cfg.OUTER_SPLITS = 5;
if cfg.FAST_MODE, cfg.OUTER_REPEATS = 1; else, cfg.OUTER_REPEATS = 2; end
cfg.INNER_SPLITS = 3;
cfg.PCA_COMPONENTS = 4;
cfg.KMEANS_CLUSTERS = 7;
cfg.KMEANS_REPLICATES = 20;
cfg.BOOTSTRAP_N = 5000;
cfg.REF_FEATURES_ORIGINAL = ["DFL","PH (cm)","SL (mm)","RNS","NGS","SGW (g)","SPW (g)","TGW (g)","NT"];
cfg.ALL_14_FEATURES_ORIGINAL = ["DFL","PH (cm)","FLA (cm2)","SL (mm)","RNS","SPW (g)","NT","TR","KR","KVZ","KLÇ","NGS","SGW (g)","TGW (g)"];
cfg.CATEGORICAL_CANDIDATES = ["TR","KR","KVZ","KLÇ"];
cfg.SAFE_ENGINEERED_CANDIDATES = ["NT_to_PH","PH_to_DFL","RNS_to_SL","FLA_to_SL","NGS_to_RNS","FLA_to_NGS","PH_x_NT","RNS_x_SL"];
cfg.ABLATION_MODELS = ["Ridge","SVR","Random Forest","XGBoost"];
cfg.CORE_MODELS = ["Ridge","SVR","Random Forest","Extra Trees","Gradient Boosting","XGBoost","CatBoost","Gaussian Process Regression","MLP","KAN"];
models = cfg.CORE_MODELS';
fullN = [10;15;12;12;12;15;15;8;10;8];
fastN = [4;4;4;4;4;4;4;3;4;3];
cfg.TRIAL_TABLE = table(models,fullN,fastN,'VariableNames',{'Model','Full','Fast'});
cfg.MLP_MAX_EPOCHS_FULL = 800;
cfg.MLP_MAX_EPOCHS_FAST = 400;
cfg.MLP_PATIENCE = 30;
cfg.KAN_MAX_EPOCHS_FULL = 140;
cfg.KAN_MAX_EPOCHS_FAST = 60;
cfg.KAN_PATIENCE_FULL = 20;
cfg.KAN_PATIENCE_FAST = 10;
cfg.AUTO_INSTALL_PYTHON = false;
end
function cfg=applyModeSettings(cfg)
if cfg.FAST_MODE, cfg.OUTER_REPEATS=1; else, cfg.OUTER_REPEATS=2; end
end
function [T,y] = loadArpaData(cfg)
assert(isfile(cfg.DATA_PATH),"Veri dosyası bulunamadı: %s",cfg.DATA_PATH);
T = readtable(cfg.DATA_PATH,'VariableNamingRule','preserve');
T.Properties.VariableNames = cellstr(strtrim(string(T.Properties.VariableNames)));
assert(any(strcmp(T.Properties.VariableNames,cfg.TARGET)),"Hedef sütun bulunamadı: %s",cfg.TARGET);
y = toNumeric(T.(cfg.TARGET));
valid = isfinite(y);
T = T(valid,:);
y = y(valid);
T = T(:,:);
assert(height(T)>=20,"Geçerli HI gözlem sayısı çok düşük.");
end
function out = checkPythonBridge()
mods = ["numpy","sklearn","xgboost","catboost"]';
avail = false(size(mods));
versions = strings(size(mods));
err = strings(size(mods));
try
    pe = pyenv;
    envText = string(pe.Status)+" | "+string(pe.Executable);
catch ME
    envText = "Python yapılandırılamadı: "+string(ME.message);
end
for i=1:numel(mods)
try
        py.importlib.import_module(char(mods(i)));
        avail(i)=true;
try
            im = py.importlib.import_module('importlib.metadata');
            pkgName = mods(i); if pkgName=="sklearn", pkgName="scikit-learn"; end
            versions(i)=string(im.version(char(pkgName)));
catch
            versions(i)="yüklü";
end
catch ME
        err(i)=string(ME.message);
end
end
out = table(mods,avail,versions,err,repmat(envText,numel(mods),1), ...
'VariableNames',{'Modul','Kullanilabilir','Surum','Hata','PyEnv'});
end
function [schemaTable,categoricalFeatures] = inspectSchema(T,cfg)
missingRequired = setdiff([cfg.ALL_14_FEATURES_ORIGINAL,cfg.TARGET],string(T.Properties.VariableNames));
if ~isempty(missingRequired), error('Beklenen sütunlar yok: %s',strjoin(missingRequired,', ')); end
categoricalFeatures = intersect(cfg.CATEGORICAL_CANDIDATES,string(T.Properties.VariableNames),'stable');
n = numel(cfg.ALL_14_FEATURES_ORIGINAL);
Sutun = strings(n,1); Veri_tipi = strings(n,1); Eksik=zeros(n,1); Tekil_deger=zeros(n,1); Ornek_degerler=strings(n,1); Onerilen_tur=strings(n,1);
for i=1:n
    name=cfg.ALL_14_FEATURES_ORIGINAL(i); col=T.(name); s=columnToStrings(col);
    Sutun(i)=name; Veri_tipi(i)=class(col); Eksik(i)=sum(isMissingLike(col));
    ss=s(~isMissingString(s)); Tekil_deger(i)=numel(unique(ss));
    u=sortCategories(unique(ss)); Ornek_degerler(i)=strjoin(u(1:min(15,numel(u))),', ');
if any(name==categoricalFeatures), Onerilen_tur(i)="Nominal kategorik"; else, Onerilen_tur(i)="Sayısal"; end
end
schemaTable=table(Sutun,Veri_tipi,Eksik,Tekil_deger,Ornek_degerler,Onerilen_tur);
end
function [audit,rawExclusions] = auditHIFormulas(T,y,cfg)
formulaNames = ["SGW/SPW","100*SGW/SPW","100*SGW/(SGW+SPW)","100*(NGS*TGW/1000)/SPW"]';
components = ["SGW (g), SPW (g)";"SGW (g), SPW (g)";"SGW (g), SPW (g)";"NGS, TGW (g), SPW (g)"];
sgw=toNumeric(T.("SGW (g)")); spw=toNumeric(T.("SPW (g)")); ngs=toNumeric(T.NGS); tgw=toNumeric(T.("TGW (g)"));
C = [safeDivide(sgw,spw),100*safeDivide(sgw,spw),100*safeDivide(sgw,sgw+spw),100*safeDivide(ngs.*tgw/1000,spw)];
rows=cell(0,8);
for j=1:size(C,2)
    x=C(:,j); valid=isfinite(x)&isfinite(y); xv=x(valid); yv=y(valid);
if numel(yv)<10 || std(xv)==0, continue; end
    r=corr(xv,yv); b=[ones(numel(xv),1),xv]\yv; yh=[ones(numel(xv),1),xv]*b;
    rmse=sqrt(mean((yv-yh).^2)); nrmse=rmse/(std(yv,0)+eps); r2=r2score(yv,yh);
    strong=abs(r)>=.98 && r2>=.95 && nrmse<=.25;
    rows(end+1,:)={formulaNames(j),components(j),r,r2,nrmse,b(2),b(1),strong};
end
if isempty(rows)
    audit=table(); rawExclusions=strings(0,1); return;
end
audit=table(string(vertcat(rows{:,1})),string(vertcat(rows{:,2})),cell2mat(rows(:,3)),cell2mat(rows(:,4)),cell2mat(rows(:,5)),cell2mat(rows(:,6)),cell2mat(rows(:,7)),logical(cell2mat(rows(:,8))), ...
'VariableNames',{'Aday_formul','Bilesenler','Pearson_r','Kalibre_R2','Kalibre_NRMSE','Egim','Sabit','Guclu_matematiksel_vekil_riski'});
audit=sortrows(audit,'Kalibre_R2','descend');
autoComponents=strings(0,1);
for i=1:height(audit)
if audit.Guclu_matematiksel_vekil_riski(i)
        autoComponents=[autoComponents; strtrim(split(audit.Bilesenler(i),','))];
end
end
autoComponents=unique(autoComponents);
switch lower(cfg.LEAKAGE_POLICY)
case "strict", rawExclusions=unique(["SGW (g)";"SPW (g)";autoComponents]);
case "audit", rawExclusions=autoComponents;
case "legacy", rawExclusions=strings(0,1);
otherwise, error('LEAKAGE_POLICY strict, audit veya legacy olmalıdır.');
end
end
function [groupColumn,groups] = inferGroupColumn(T,cfg)
if strlength(cfg.MANUAL_GROUP_COLUMN)>0
    assert(any(strcmp(T.Properties.VariableNames,cfg.MANUAL_GROUP_COLUMN)),"MANUAL_GROUP_COLUMN bulunamadı.");
    groupColumn=cfg.MANUAL_GROUP_COLUMN;
else
    preferred=["OTGB NO","OTGB_NO","Genotype","Genotip","Line","Hat","Accession"];
    groupColumn="";
for name=preferred
if any(strcmp(T.Properties.VariableNames,name))
            g=columnToStrings(T.(name)); nu=numel(unique(g));
if nu>=cfg.OUTER_SPLITS && nu<height(T), groupColumn=name; break; end
end
end
end
if strlength(groupColumn)==0, groups=strings(0,1); else, groups=columnToStrings(T.(groupColumn)); groups(isMissingString(groups))="__MISSING_GROUP__"; end
end
function specs = buildFeatureSpecs(cfg,rawExclusions)
ref=setdiff(cfg.REF_FEATURES_ORIGINAL,rawExclusions,'stable'); all14=setdiff(cfg.ALL_14_FEATURES_ORIGINAL,rawExclusions,'stable');
specs=struct();
specs.A0_REF9_SAFE=makeSpec(ref,false,false,"Referans özellikler; riskli ham bileşenler çıkarıldı");
specs.A1_ALL14_SAFE=makeSpec(all14,false,false,"14 temel özellik; riskli ham bileşenler çıkarıldı");
specs.A2_SAFE_ENGINEERED=makeSpec(all14,true,false,"A1 + güvenli agronomik özellik mühendisliği");
specs.A3_SAFE_ENGINEERED_PCAKMEANS=makeSpec(all14,true,true,"A2 + fold-içi PCA ve K-Means özellikleri");
end
function s=makeSpec(base,eng,pk,desc), s=struct('base_features',base,'add_engineered',eng,'add_pca_kmeans',pk,'description',string(desc)); end
function tbl=featureSpecsToTable(specs)
keys=string(fieldnames(specs)); n=numel(keys); base=strings(n,1); eng=false(n,1); pk=false(n,1); desc=strings(n,1);
for i=1:n, s=specs.(keys(i)); base(i)=strjoin(s.base_features,', '); eng(i)=s.add_engineered; pk(i)=s.add_pca_kmeans; desc(i)=s.description; end
tbl=table(keys,base,eng,pk,desc,'VariableNames',{'Feature_Set','Base_Features','Engineered','PCA_KMeans','Aciklama'});
end
function names = availableEngineered(base)
req = { ...
"NT_to_PH",   ["NT","PH (cm)"]; ...
"PH_to_DFL",  ["PH (cm)","DFL"]; ...
"RNS_to_SL",  ["RNS","SL (mm)"]; ...
"FLA_to_SL",  ["FLA (cm2)","SL (mm)"]; ...
"NGS_to_RNS", ["NGS","RNS"]; ...
"FLA_to_NGS", ["FLA (cm2)","NGS"]; ...
"PH_x_NT",    ["PH (cm)","NT"]; ...
"RNS_x_SL",   ["RNS","SL (mm)"] ...
    };
base = string(base(:));
names = strings(0,1);
for i = 1:size(req,1)
    required = string(req{i,2}(:));
if all(ismember(required,base))
        names(end+1,1) = string(req{i,1});
end
end
end
function tbl=modelImplementationTable()
Model=["Ridge";"SVR";"Random Forest";"Extra Trees";"Gradient Boosting";"XGBoost";"CatBoost";"Gaussian Process Regression";"MLP";"KAN"];
Uygulama=["MATLAB kapalı-form Ridge";"fitrsvm";"fitrensemble Bag";"Python scikit-learn ExtraTreesRegressor";"fitrensemble LSBoost";"Python xgboost.XGBRegressor";"Python catboost.CatBoostRegressor";"fitrgp Matern 5/2";"Saf MATLAB ReLU-Adam MLP";"Saf MATLAB SiLU+B-spline KAN + AdamW"];
Not=["alpha aynı aralık";"RBF gamma→KernelScale dönüşümü";"bootstrap bagging + predictor subsampling";"algoritmayı aynen korumak için Python";"stochastic LSBoost";"algoritmayı aynen korumak için Python";"algoritmayı aynen korumak için Python";"fixed hyperparameters; optimizer yok";"%15 validation ve early stopping";"tek KANLinear eşdeğeri"];
tbl=table(Model,Uygulama,Not);
end
function tbl=hyperparameterSpaceTable(cfg)
Model=cfg.CORE_MODELS';
Uzay=["alpha log[1e-4,1e3]";"C log[1e-2,1e3]; epsilon log[1e-3,.5]; gamma log[1e-4,1]";"trees 200:50:500; depth {None,4,6,10,16}; leaf 1:6; max_features [.4,1]";"RF ile aynı";"trees 100:50:400; lr log[.01,.15]; depth 1:4; leaf 2:10; subsample [.6,1]";"trees 100:50:450; lr; depth 2:6; min_child 1:8; subsample; colsample; alpha; lambda";"iterations 100:50:450; lr; depth 3:7; l2; random_strength";"alpha; constant; length; white_noise (log aralıklar)";"hidden {16,32,64,32-16,64-32}; alpha; lr; batch {16,32,64}";"grid 3:7; order 2:3; lr; weight_decay"];
Trials=zeros(numel(Model),1); for i=1:numel(Model), Trials(i)=getTrialCount(cfg,Model(i)); end
tbl=table(Model,Uzay,Trials);
end
function n=getTrialCount(cfg,modelName)
idx=find(cfg.TRIAL_TABLE.Model==string(modelName),1); assert(~isempty(idx),'Model trial sayısı bulunamadı.'); if cfg.FAST_MODE, n=cfg.TRIAL_TABLE.Fast(idx); else, n=cfg.TRIAL_TABLE.Full(idx); end
end
function splits=makeOuterSplits(n,groups,cfg)
if ~isempty(groups)
    splits=makeGroupKFold(groups,min(cfg.OUTER_SPLITS,numel(unique(groups))));
if numel(splits)<3, error('GroupKFold için en az 3 grup/fold gereklidir.'); end
else
    splits=struct('trainIdx',{},'testIdx',{});
for r=1:cfg.OUTER_REPEATS
        rng(cfg.SEED+r-1,'twister'); cv=cvpartition(n,'KFold',cfg.OUTER_SPLITS);
for k=1:cfg.OUTER_SPLITS
            splits(end+1).trainIdx=find(training(cv,k));
            splits(end).testIdx=find(test(cv,k));
end
end
end
end
function splits=makeInnerSplits(n,groups,k,seed)
if ~isempty(groups), splits=makeGroupKFold(groups,min(k,numel(unique(groups))));
else
    rng(seed,'twister'); cv=cvpartition(n,'KFold',k); splits=struct('trainIdx',{},'testIdx',{});
for j=1:k, splits(end+1).trainIdx=find(training(cv,j)); splits(end).testIdx=find(test(cv,j)); end
end
end
function splits=makeGroupKFold(groups,k)
g=string(groups(:)); ug=unique(g,'stable'); counts=zeros(numel(ug),1); for i=1:numel(ug), counts(i)=sum(g==ug(i)); end
[~,order]=sort(counts,'descend'); foldLoad=zeros(k,1); foldOfGroup=zeros(numel(ug),1);
for ii=1:numel(order), gi=order(ii); [~,f]=min(foldLoad); foldOfGroup(gi)=f; foldLoad(f)=foldLoad(f)+counts(gi); end
splits=struct('trainIdx',{},'testIdx',{});
for f=1:k, testGroups=ug(foldOfGroup==f); isTest=ismember(g,testGroups); splits(end+1).trainIdx=find(~isTest); splits(end).testIdx=find(isTest); end
end
function tbl=summarizeOuterSplits(splits,groups)
n=numel(splits); Fold=(1:n)'; TrainN=zeros(n,1); TestN=zeros(n,1); GroupOverlap=strings(n,1);
for i=1:n
    TrainN(i)=numel(splits(i).trainIdx); TestN(i)=numel(splits(i).testIdx);
if isempty(groups), GroupOverlap(i)="-"; else, GroupOverlap(i)=string(numel(intersect(unique(groups(splits(i).trainIdx)),unique(groups(splits(i).testIdx))))); end
end
tbl=table(Fold,TrainN,TestN,GroupOverlap);
end
function [resultTag,fingerprint]=makeResultTag(cfg,rawExclusions,groupColumn)
if cfg.FAST_MODE, mode="fast"; else, mode="full"; end
trialText=strjoin(cfg.TRIAL_TABLE.Model+":"+string(cfg.TRIAL_TABLE.Full)+"/"+string(cfg.TRIAL_TABLE.Fast),"|");
payload=cfg.CODE_VERSION+"|"+mode+"|"+string(cfg.SEED)+"|"+cfg.LEAKAGE_POLICY+"|"+strjoin(rawExclusions,",")+"|"+groupColumn+"|"+string(cfg.OUTER_SPLITS)+"|"+string(cfg.OUTER_REPEATS)+"|"+string(cfg.INNER_SPLITS)+"|"+trialText;
fingerprint=simpleHash(payload); resultTag=mode+"_"+fingerprint;
end
function h=simpleHash(txt)
b=uint8(unicode2native(char(txt),'UTF-8')); x=uint32(2166136261); prime=uint64(16777619); modv=uint64(4294967296);
for i=1:numel(b), x=bitxor(x,uint32(b(i))); x=uint32(mod(uint64(x)*prime,modv)); end
h=lower(string(dec2hex(x,8)));
end
function [foldResults,predictions]=runOrLoadNested(stage,featureKey,modelName,nTrials,specs,T,y,groups,outerSplits,resultTag,cfg)
[resultPath,predPath]=checkpointPaths(stage,featureKey,modelName,resultTag,cfg);
if cfg.RESUME_FROM_CHECKPOINT && isfile(resultPath) && isfile(predPath)
    fr=readtable(resultPath,'TextType','string'); pr=readtable(predPath,'TextType','string');
if any(strcmp(fr.Properties.VariableNames,'outer_fold')) && numel(unique(fr.outer_fold))==numel(outerSplits)
        fprintf('[Checkpoint] Yüklendi: %s | %s\n',featureKey,modelName); foldResults=fr; predictions=pr; return;
end
end
spec=specs.(featureKey);
[foldResults,predictions]=nestedCVEvaluate(modelName,featureKey,spec,T,y,groups,outerSplits,nTrials,stage,cfg);
writetable(foldResults,resultPath); writetable(predictions,predPath);
end
function [rp,pp]=checkpointPaths(stage,featureKey,modelName,resultTag,cfg)
stem=safeFilename(stage)+"_"+safeFilename(featureKey)+"_"+safeFilename(modelName)+"_"+resultTag;
rp=fullfile(cfg.OUTPUT_DIR,stem+"_fold_results.csv"); pp=fullfile(cfg.OUTPUT_DIR,stem+"_predictions.csv");
end
function s=safeFilename(x), s=regexprep(string(x),'[^A-Za-z0-9]+','_'); s=strip(s,'both','_'); end
function [foldResults,predictions]=nestedCVEvaluate(modelName,featureKey,spec,T,y,groups,outerSplits,nTrials,stage,cfg)
frCells=cell(numel(outerSplits),1); prCells=cell(numel(outerSplits),1);
for f=1:numel(outerSplits)
    t0=tic; tr=outerSplits(f).trainIdx; te=outerSplits(f).testIdx;
    Ttr=T(tr,:); Tte=T(te,:); ytr=y(tr); yte=y(te);
if isempty(groups), gtr=strings(0,1); else, gtr=groups(tr); end
    foldSeed=cfg.SEED+1000*f+stableNameSeed(modelName)+stableNameSeed(featureKey);
    fprintf('[NestedCV] %s | %s | fold %d/%d | %d trial\n',stage,modelName,f,numel(outerSplits),nTrials);
    [bestParams,bestInnerRMSE]=optimizeOnOuterTrain(modelName,spec,Ttr,ytr,gtr,foldSeed,nTrials,cfg);
    [Xtr,Xte,transformInfo]=fitAndApplyFoldPipeline(Ttr,Tte,spec,cfg,foldSeed);
    [ys,yscaler]=fitTargetScaler(ytr);
    mdl=trainModelCore(modelName,Xtr,ys,bestParams,foldSeed,cfg);
    predTr=inverseTargetScale(predictModelCore(mdl,Xtr),yscaler);
    predTe=inverseTargetScale(predictModelCore(mdl,Xte),yscaler);
    [trR2,~,~]=metrics3(ytr,predTr); [teR2,teRMSE,teMAE]=metrics3(yte,predTe);
    elapsed=toc(t0);
    frCells{f}=table(f,string(modelName),string(featureKey),trR2,teR2,teRMSE,teMAE,bestInnerRMSE,elapsed,string(jsonencode(bestParams)), ...
'VariableNames',{'outer_fold','model','feature_set','train_r2','test_r2','test_rmse','test_mae','best_inner_rmse','elapsed_seconds','best_params'});
    nte=numel(te); prCells{f}=table(repmat(f,nte,1),repmat(string(modelName),nte,1),repmat(string(featureKey),nte,1),te(:),yte(:),predTe(:),yte(:)-predTe(:), ...
'VariableNames',{'outer_fold','model','feature_set','row_index','y_true','y_pred','residual'});
end
foldResults=vertcat(frCells{:}); predictions=vertcat(prCells{:});
end
function [bestParams,bestScore]=optimizeOnOuterTrain(modelName,spec,Ttr,ytr,gtr,foldSeed,nTrials,cfg)
inner=makeInnerSplits(height(Ttr),gtr,cfg.INNER_SPLITS,foldSeed+17); bestScore=inf; bestParams=struct();
for trial=1:nTrials
    params=sampleHyperparams(modelName,foldSeed+10000*trial); scores=zeros(numel(inner),1);
for j=1:numel(inner)
        it=inner(j).trainIdx; iv=inner(j).testIdx;
        [Xi,Xv]=fitAndApplyFoldPipeline(Ttr(it,:),Ttr(iv,:),spec,cfg,foldSeed+trial*100+j);
        [ysi,yscaler]=fitTargetScaler(ytr(it));
        mdl=trainModelCore(modelName,Xi,ysi,params,foldSeed+trial*100+j,cfg);
        pv=inverseTargetScale(predictModelCore(mdl,Xv),yscaler);
        [~,scores(j),~]=metrics3(ytr(iv),pv);
end
    score=mean(scores);
if score<bestScore, bestScore=score; bestParams=params; end
end
end
function p=sampleHyperparams(modelName,seed)
rng(seed,'twister'); name=string(modelName); p=struct();
switch name
case "Ridge"
        p.alpha=logUniform(1e-4,1e3);
case "SVR"
        p.C=logUniform(1e-2,1e3); p.epsilon=logUniform(1e-3,.5); p.gamma=logUniform(1e-4,1);
case {"Random Forest","Extra Trees"}
        p.n_estimators=sampleStep(200,500,50); depths=[0 4 6 10 16]; p.max_depth=depths(randi(numel(depths))); p.min_samples_leaf=randi([1 6]); p.max_features=.4+.6*rand;
case "Gradient Boosting"
        p.n_estimators=sampleStep(100,400,50); p.learning_rate=logUniform(.01,.15); p.max_depth=randi([1 4]); p.min_samples_leaf=randi([2 10]); p.subsample=.6+.4*rand;
case "XGBoost"
        p.n_estimators=sampleStep(100,450,50); p.learning_rate=logUniform(.01,.15); p.max_depth=randi([2 6]); p.min_child_weight=randi([1 8]); p.subsample=.6+.4*rand; p.colsample_bytree=.5+.5*rand; p.reg_alpha=logUniform(1e-5,3); p.reg_lambda=logUniform(1e-3,20);
case "CatBoost"
        p.iterations=sampleStep(100,450,50); p.learning_rate=logUniform(.01,.15); p.depth=randi([3 7]); p.l2_leaf_reg=logUniform(1e-2,20); p.random_strength=logUniform(1e-3,5);
case "Gaussian Process Regression"
        p.alpha=logUniform(1e-8,1e-2); p.kernel_constant=logUniform(1e-2,1e2); p.length_scale=logUniform(1e-2,1e2); p.white_noise=logUniform(1e-4,2);
case "MLP"
        arch={[16],[32],[64],[32 16],[64 32]}; p.hidden_layer_sizes=arch{randi(5)}; p.alpha=logUniform(1e-6,1e-1); p.learning_rate_init=logUniform(1e-4,1e-2); bs=[16 32 64]; p.batch_size=bs(randi(3));
case "KAN"
        p.grid_size=randi([3 7]); p.spline_order=randi([2 3]); p.lr=logUniform(3e-4,5e-3); p.weight_decay=logUniform(1e-6,1e-2); p.batch_size=128;
otherwise, error('Tanımsız model: %s',name);
end
end
function x=logUniform(a,b), x=exp(log(a)+(log(b)-log(a))*rand); end
function x=sampleStep(a,b,s), vals=a:s:b; x=vals(randi(numel(vals))); end
function [Xtr,Xte,info]=fitAndApplyFoldPipeline(Ttr,Tte,spec,cfg,seed)
[Ftr,numNames,catNames]=buildAgronomicFeatures(Ttr,spec,cfg); [Fte,~,~]=buildAgronomicFeatures(Tte,spec,cfg);
[prep,Xtr]=fitPreprocessor(Ftr,numNames,catNames); Xte=applyPreprocessor(prep,Fte);
info=struct('preprocessor',prep);
if spec.add_pca_kmeans
    [augment,Xtr]=fitPCAKMeans(Xtr,cfg,seed); Xte=applyPCAKMeans(augment,Xte); info.augment=augment;
end
end
function [F,numNames,catNames]=buildAgronomicFeatures(T,spec,cfg)
base = string(spec.base_features(:));
F = T(:,cellstr(base));
catNames = intersect( ...
    base, ...
    string(cfg.CATEGORICAL_CANDIDATES(:)), ...
'stable');
numNames = setdiff(base,catNames,'stable');
for k = 1:numel(numNames)
    name = numNames(k);
    F.(name) = toNumeric(F.(name));
end
if spec.add_engineered
    eng = availableEngineered(base);
for k = 1:numel(eng)
        name = eng(k);
v = nan(height(F),1);
switch name
case "NT_to_PH"
                v = safeDivide( ...
                    toNumeric(F.NT), ...
                    toNumeric(F.("PH (cm)")) );
case "PH_to_DFL"
                v = safeDivide( ...
                    toNumeric(F.("PH (cm)")), ...
                    toNumeric(F.DFL) );
case "RNS_to_SL"
                v = safeDivide( ...
                    toNumeric(F.RNS), ...
                    toNumeric(F.("SL (mm)")) );
case "FLA_to_SL"
                v = safeDivide( ...
                    toNumeric(F.("FLA (cm2)")), ...
                    toNumeric(F.("SL (mm)")) );
case "NGS_to_RNS"
                v = safeDivide( ...
                    toNumeric(F.NGS), ...
                    toNumeric(F.RNS) );
case "FLA_to_NGS"
                v = safeDivide( ...
                    toNumeric(F.("FLA (cm2)")), ...
                    toNumeric(F.NGS) );
case "PH_x_NT"
                v = ...
                    toNumeric(F.("PH (cm)")) .* ...
                    toNumeric(F.NT);
case "RNS_x_SL"
                v = ...
                    toNumeric(F.RNS) .* ...
                    toNumeric(F.("SL (mm)"));
otherwise
                error( ...
"Tanımlanmamış engineered feature: %s", ...
                    name);
end
        v(~isfinite(v)) = NaN;
        F.(name) = v;
        numNames(end+1,1) = name;
end
end
end
function [prep,X]=fitPreprocessor(F,numNames,catNames)
prep=struct(); prep.numNames=numNames; prep.catNames=catNames; X=[];
prep.numMedian=zeros(numel(numNames),1); prep.numMean=zeros(numel(numNames),1); prep.numStd=ones(numel(numNames),1);
for i=1:numel(numNames)
    v=toNumeric(F.(numNames(i))); med=median(v,'omitnan'); if isnan(med), med=0; end; v(~isfinite(v))=med;
    mu=mean(v); sd=std(v,1); if ~isfinite(sd)||sd<eps, sd=1; end
    prep.numMedian(i)=med; prep.numMean(i)=mu; prep.numStd(i)=sd; X=[X,(v-mu)/sd];
end
prep.catMode=cell(numel(catNames),1); prep.catLevels=cell(numel(catNames),1);
for i=1:numel(catNames)
    s=columnToStrings(F.(catNames(i))); miss=isMissingString(s); modeVal=mostFrequentCategory(s(~miss)); if strlength(modeVal)==0, modeVal="__MISSING__"; end; s(miss)=modeVal;
    lev=sortCategories(unique(s)); prep.catMode{i}=modeVal; prep.catLevels{i}=lev;
for j=1:numel(lev), X=[X,double(s==lev(j))]; end
end
end
function X=applyPreprocessor(prep,F)
X=[];
for i=1:numel(prep.numNames), v=toNumeric(F.(prep.numNames(i))); v(~isfinite(v))=prep.numMedian(i); X=[X,(v-prep.numMean(i))/prep.numStd(i)]; end
for i=1:numel(prep.catNames), s=columnToStrings(F.(prep.catNames(i))); s(isMissingString(s))=prep.catMode{i}; lev=prep.catLevels{i}; for j=1:numel(lev), X=[X,double(s==lev(j))]; end, end
end
function [aug,Xout]=fitPCAKMeans(X,cfg,seed)
nComp=max(1,min([cfg.PCA_COMPONENTS,size(X,2),size(X,1)-1])); [coeff,score,~,~,~,mu]=pca(X,'NumComponents',nComp);
nClust=max(2,min(cfg.KMEANS_CLUSTERS,size(X,1)-1)); rng(seed,'twister'); [idx,C]=kmeans(X,nClust,'Replicates',cfg.KMEANS_REPLICATES,'Start','plus','Display','off');
clusterOHE=zeros(size(X,1),nClust); clusterOHE(sub2ind(size(clusterOHE),(1:size(X,1))',idx))=1;
Xout=[X,score,clusterOHE]; aug=struct('coeff',coeff,'mu',mu,'centers',C,'nClusters',nClust);
end
function Xout=applyPCAKMeans(aug,X)
pcs=(X-aug.mu)*aug.coeff; D=pdist2(X,aug.centers,'squaredeuclidean'); [~,idx]=min(D,[],2); ohe=zeros(size(X,1),aug.nClusters); ohe(sub2ind(size(ohe),(1:size(X,1))',idx))=1; Xout=[X,pcs,ohe];
end
function [ys,s]=fitTargetScaler(y), s.mu=mean(y); s.sd=std(y,1); if s.sd<eps, s.sd=1; end; ys=(y-s.mu)/s.sd; end
function y=inverseTargetScale(ys,s), y=ys*s.sd+s.mu; end
function mdl=trainModelCore(modelName,X,y,p,seed,cfg)
rng(seed,'twister'); name=string(modelName); mdl=struct('kind',name);
switch name
case "Ridge"
        Xa=[ones(size(X,1),1),X]; P=diag([0;p.alpha*ones(size(X,2),1)]); mdl.beta=(Xa'*Xa+P)\\(Xa'*y);
case "SVR"
        ks=1/sqrt(2*p.gamma); mdl.obj=fitrsvm(X,y,'KernelFunction','gaussian','BoxConstraint',p.C,'Epsilon',p.epsilon,'KernelScale',ks,'Standardize',false);
case "Random Forest"
        maxSplits=depthToSplits(p.max_depth,size(X,1)); nvars=max(1,min(size(X,2),round(p.max_features*size(X,2))));
        t=templateTree('Type','regression','MinLeafSize',p.min_samples_leaf,'MaxNumSplits',maxSplits,'NumVariablesToSample',nvars,'Reproducible',true);
        mdl.obj=fitrensemble(X,y,'Method','Bag','NumLearningCycles',p.n_estimators,'Learners',t,'Resample','on','FResample',1,'Replace','on');
case "Extra Trees"
        mdl.obj=trainPythonRegressor(name,X,y,p,seed);
case "Gradient Boosting"
        maxSplits=depthToSplits(p.max_depth,size(X,1)); t=templateTree('Type','regression','MinLeafSize',p.min_samples_leaf,'MaxNumSplits',maxSplits,'NumVariablesToSample','all');
        mdl.obj=fitrensemble(X,y,'Method','LSBoost','NumLearningCycles',p.n_estimators,'LearnRate',p.learning_rate,'Learners',t,'Resample','on','FResample',p.subsample,'Replace','off');
case "XGBoost"
        mdl.obj=trainPythonRegressor(name,X,y,p,seed);
case "CatBoost"
        mdl.obj=trainPythonRegressor(name,X,y,p,seed);
case "Gaussian Process Regression"
        sigma=sqrt(max(p.white_noise+p.alpha,1e-12)); kp=[p.length_scale;sqrt(p.kernel_constant)];
        mdl.obj=fitrgp(X,y,'KernelFunction','matern52','KernelParameters',kp,'Sigma',sigma,'ConstantSigma',true,'BasisFunction','none','FitMethod','none','PredictMethod','exact','Standardize',false);
case "MLP"
        mdl.obj=trainMLP(X,y,p,seed,cfg);
case "KAN"
        mdl.obj=trainKAN(X,y,p,seed,cfg);
otherwise, error('Tanımsız model: %s',name);
end
end
function yhat=predictModelCore(mdl,X)
switch mdl.kind
case "Ridge", yhat=[ones(size(X,1),1),X]*mdl.beta;
case {"SVR","Random Forest","Gradient Boosting","Gaussian Process Regression"}, yhat=predict(mdl.obj,X);
case {"Extra Trees","XGBoost","CatBoost"}, yhat=predictPythonRegressor(mdl.obj,X);
case "MLP", yhat=predictMLP(mdl.obj,X);
case "KAN", yhat=predictKAN(mdl.obj,X);
end
yhat=double(yhat(:));
end
function m=depthToSplits(depth,n), if depth==0, m=max(1,n-1); else, m=min(max(1,n-1),2^depth-1); end, end
function obj=trainPythonRegressor(name,X,y,p,seed)
assertPythonModules(name); np=py.importlib.import_module('numpy'); pyX=np.asarray(X,pyargs('dtype','float64')); pyY=np.asarray(y(:),pyargs('dtype','float64')); pyY=pyY.reshape(int64(numel(y)));
switch name
case "Extra Trees"
        ens=py.importlib.import_module('sklearn.ensemble'); if p.max_depth==0, md=py.None; else, md=int64(p.max_depth); end
        obj=ens.ExtraTreesRegressor(pyargs('n_estimators',int64(p.n_estimators),'max_depth',md,'min_samples_leaf',int64(p.min_samples_leaf),'max_features',p.max_features,'random_state',int64(seed),'n_jobs',int64(2)));
case "XGBoost"
        xm=py.importlib.import_module('xgboost'); obj=xm.XGBRegressor(pyargs('objective','reg:squarederror','n_estimators',int64(p.n_estimators),'learning_rate',p.learning_rate,'max_depth',int64(p.max_depth),'min_child_weight',p.min_child_weight,'subsample',p.subsample,'colsample_bytree',p.colsample_bytree,'reg_alpha',p.reg_alpha,'reg_lambda',p.reg_lambda,'random_state',int64(seed),'n_jobs',int64(2),'verbosity',int64(0),'tree_method','hist'));
case "CatBoost"
        cm=py.importlib.import_module('catboost'); obj=cm.CatBoostRegressor(pyargs('iterations',int64(p.iterations),'learning_rate',p.learning_rate,'depth',int64(p.depth),'l2_leaf_reg',p.l2_leaf_reg,'random_strength',p.random_strength,'loss_function','RMSE','random_seed',int64(seed),'verbose',int64(0),'allow_writing_files',false,'thread_count',int64(2)));
end
obj.fit(pyX,pyY);
end
function y=predictPythonRegressor(obj,X)
np=py.importlib.import_module('numpy'); arr=obj.predict(np.asarray(X,pyargs('dtype','float64'))); y=double(np.asarray(arr,pyargs('dtype','float64'))); y=y(:);
end
function assertPythonModules(modelName)
req="numpy"; if modelName=="Extra Trees", req=[req,"sklearn"]; elseif modelName=="XGBoost", req=[req,"xgboost"]; elseif modelName=="CatBoost", req=[req,"catboost"]; end
for m=req, try, py.importlib.import_module(char(m)); catch ME, error('Python modülü gerekli (%s) fakat yüklenemedi. MATLAB pyenv ayarını yapın ve paketi kurun. Ayrıntı: %s',m,ME.message); end, end
end
function mdl=trainMLP(X,y,p,seed,cfg)
rng(seed,'twister'); [tr,va]=trainValIndices(size(X,1),.15,seed); dims=[size(X,2),p.hidden_layer_sizes,1]; L=numel(dims)-1;
W=cell(L,1); b=cell(L,1); mW=cell(L,1); vW=cell(L,1); mb=cell(L,1); vb=cell(L,1);
for l=1:L, bound=sqrt(6/(dims(l)+dims(l+1))); W{l}=(2*rand(dims(l),dims(l+1))-1)*bound; b{l}=zeros(1,dims(l+1)); mW{l}=zeros(size(W{l})); vW{l}=zeros(size(W{l})); mb{l}=zeros(size(b{l})); vb{l}=zeros(size(b{l})); end
if cfg.FAST_MODE, maxEpoch=cfg.MLP_MAX_EPOCHS_FAST; else, maxEpoch=cfg.MLP_MAX_EPOCHS_FULL; end
beta1=.9; beta2=.999; ae=1e-8; tstep=0; bestScore=-inf; wait=0; bestW=W; bestB=b; nTrain=numel(tr);
for epoch=1:maxEpoch
    ord=tr(randperm(nTrain)); bs=min(p.batch_size,nTrain);
for st=1:bs:nTrain
        id=ord(st:min(st+bs-1,nTrain)); [yh,A,Z]=mlpForward(X(id,:),W,b); delta=2*(yh-y(id))/numel(id); gW=cell(L,1); gb=cell(L,1);
for l=L:-1:1
            gW{l}=A{l}'*delta + (p.alpha/nTrain)*W{l}; gb{l}=sum(delta,1);
if l>1, delta=(delta*W{l}').*(Z{l-1}>0); end
end
        tstep=tstep+1;
for l=1:L
            mW{l}=beta1*mW{l}+(1-beta1)*gW{l}; vW{l}=beta2*vW{l}+(1-beta2)*(gW{l}.^2); mb{l}=beta1*mb{l}+(1-beta1)*gb{l}; vb{l}=beta2*vb{l}+(1-beta2)*(gb{l}.^2);
            W{l}=W{l}-p.learning_rate_init*(mW{l}/(1-beta1^tstep))./(sqrt(vW{l}/(1-beta2^tstep))+ae); b{l}=b{l}-p.learning_rate_init*(mb{l}/(1-beta1^tstep))./(sqrt(vb{l}/(1-beta2^tstep))+ae);
end
end
    pv=mlpPredictRaw(X(va,:),W,b); sc=r2score(y(va),pv); if ~isfinite(sc), sc=-mean((y(va)-pv).^2); end
if sc>bestScore+1e-4, bestScore=sc; bestW=W; bestB=b; wait=0; else, wait=wait+1; end
if wait>=cfg.MLP_PATIENCE, break; end
end
mdl=struct('W',{bestW},'b',{bestB},'epochs',epoch,'bestValidationScore',bestScore);
end
function [yh,A,Z]=mlpForward(X,W,b)
L=numel(W); A=cell(L,1); Z=cell(max(0,L-1),1); A{1}=X;
for l=1:L
    z=A{l}*W{l}+b{l};
if l<L, Z{l}=z; if l+1<=L, A{l+1}=max(z,0); end, else, yh=z; end
end
end
function y=mlpPredictRaw(X,W,b)
A=X; for l=1:numel(W), Z=A*W{l}+b{l}; if l<numel(W), A=max(Z,0); else, y=Z; end, end; y=y(:);
end
function y=predictMLP(mdl,X), y=mlpPredictRaw(X,mdl.W,mdl.b); end
function mdl=trainKAN(X,y,p,seed,cfg)
rng(seed,'twister'); Phi=kanDesign(X,p.grid_size,p.spline_order); d=size(X,2); nBasis=p.grid_size+p.spline_order; w=[(2*rand(d,1)-1)/sqrt(d); .02*randn(d*nBasis,1)];
[tr,va]=trainValIndices(size(X,1),.15,seed); if cfg.FAST_MODE, maxEpoch=cfg.KAN_MAX_EPOCHS_FAST; patience=cfg.KAN_PATIENCE_FAST; else, maxEpoch=cfg.KAN_MAX_EPOCHS_FULL; patience=cfg.KAN_PATIENCE_FULL; end
m=zeros(size(w)); v=zeros(size(w)); beta1=.9; beta2=.999; ae=1e-8; tstep=0; bestLoss=inf; bestW=w; wait=0; nTrain=numel(tr); bs=min(p.batch_size,nTrain);
for epoch=1:maxEpoch
    ord=tr(randperm(nTrain));
for st=1:bs:nTrain
        id=ord(st:min(st+bs-1,nTrain)); Pb=Phi(id,:); grad=2*(Pb'*(Pb*w-y(id)))/numel(id); gn=norm(grad); if gn>5, grad=grad*(5/gn); end
        tstep=tstep+1; m=beta1*m+(1-beta1)*grad; v=beta2*v+(1-beta2)*(grad.^2); mhat=m/(1-beta1^tstep); vhat=v/(1-beta2^tstep);
        w=w*(1-p.lr*p.weight_decay)-p.lr*mhat./(sqrt(vhat)+ae);
end
    pv=Phi(va,:)*w; vl=mean((pv-y(va)).^2); if vl<bestLoss-1e-6, bestLoss=vl; bestW=w; wait=0; else, wait=wait+1; end; if wait>=patience, break; end
end
mdl=struct('w',bestW,'grid_size',p.grid_size,'spline_order',p.spline_order,'epochs',epoch,'bestValidationLoss',bestLoss);
end
function y=predictKAN(mdl,X), y=kanDesign(X,mdl.grid_size,mdl.spline_order)*mdl.w; y=y(:); end
function Phi=kanDesign(X,gridSize,order)
base=X./(1+exp(-X)); n=size(X,1); d=size(X,2); h=6/gridSize; grid=(-order:(gridSize+order))*h-3; nb=gridSize+order; spl=zeros(n,d*nb);
for q=1:d
    x=X(:,q); B=zeros(n,numel(grid)-1); for j=1:size(B,2), B(:,j)=double(x>=grid(j) & x<grid(j+1)); end
for k=1:order
        Bn=zeros(n,size(B,2)-1); for j=1:size(Bn,2), ld=max(grid(j+k)-grid(j),1e-12); rd=max(grid(j+k+1)-grid(j+1),1e-12); Bn(:,j)=((x-grid(j))/ld).*B(:,j)+((grid(j+k+1)-x)/rd).*B(:,j+1); end; B=Bn;
end
    spl(:,(q-1)*nb+(1:nb))=B;
end
Phi=[base,spl];
end
function [tr,va]=trainValIndices(n,fraction,seed)
rng(seed,'twister'); ord=randperm(n); nv=max(1,ceil(n*fraction)); va=ord(1:nv)'; tr=ord(nv+1:end)'; if numel(tr)<5, error('Eğitim alt kümesi çok küçük.'); end
end
function tbl=summarizeFoldResults(F)
if isempty(F), tbl=table(); return; end
F.model=string(F.model); F.feature_set=string(F.feature_set); fs=unique(F.feature_set,'stable'); models=unique(F.model,'stable'); rows={};
for a=1:numel(fs), for b=1:numel(models), P=F(F.feature_set==fs(a)&F.model==models(b),:); if isempty(P), continue; end; rows(end+1,:)={fs(a),models(b),mean(P.test_r2),std(P.test_r2,0),mean(P.test_rmse),std(P.test_rmse,0),mean(P.test_mae),std(P.test_mae,0),mean(P.train_r2),numel(unique(P.outer_fold)),sum(P.elapsed_seconds)}; end, end
tbl=table(string(vertcat(rows{:,1})),string(vertcat(rows{:,2})),cell2mat(rows(:,3)),cell2mat(rows(:,4)),cell2mat(rows(:,5)),cell2mat(rows(:,6)),cell2mat(rows(:,7)),cell2mat(rows(:,8)),cell2mat(rows(:,9)),cell2mat(rows(:,10)),cell2mat(rows(:,11)), ...
'VariableNames',{'feature_set','model','R2_mean','R2_sd','RMSE_mean','RMSE_sd','MAE_mean','MAE_sd','Train_R2_mean','Fold_count','Time_seconds'}); tbl.Overfit_gap=tbl.Train_R2_mean-tbl.R2_mean; tbl=sortrows(tbl,{'feature_set','R2_mean'},{'ascend','descend'});
end
function tbl=ablationPairedTests(F,cfg)
pairs=["A0_REF9_SAFE","A1_ALL14_SAFE";"A1_ALL14_SAFE","A2_SAFE_ENGINEERED";"A2_SAFE_ENGINEERED","A3_SAFE_ENGINEERED_PCAKMEANS"]; rows={};
for m=cfg.ABLATION_MODELS
    M=F(string(F.model)==m,:);
for j=1:size(pairs,1)
        L=M(string(M.feature_set)==pairs(j,1),:); R=M(string(M.feature_set)==pairs(j,2),:); [common,ia,ib]=intersect(L.outer_fold,R.outer_fold); d=R.test_r2(ib)-L.test_r2(ia);
if numel(d)<3 || all(abs(d)<1e-12), stat=0; p=1; else, [p,~,st]=signrank(R.test_r2(ib),L.test_r2(ia)); stat=st.signedrank; end
        rows(end+1,:)={m,pairs(j,1),pairs(j,2),mean(d),median(d),stat,p,numel(common)};
end
end
tbl=table(string(vertcat(rows{:,1})),string(vertcat(rows{:,2})),string(vertcat(rows{:,3})),cell2mat(rows(:,4)),cell2mat(rows(:,5)),cell2mat(rows(:,6)),cell2mat(rows(:,7)),cell2mat(rows(:,8)), ...
'VariableNames',{'model','left_feature_set','right_feature_set','mean_R2_change_right_minus_left','median_R2_change','wilcoxon_statistic','p_raw','n_paired_folds'}); tbl.p_holm=holmAdjust(tbl.p_raw); tbl.significant_holm_0_05=tbl.p_holm<.05;
end
function f=plotAblation(S,specs,cfg)
keys=string(fieldnames(specs)); f=figure('Name','Ablation'); hold on;
for m=cfg.ABLATION_MODELS
    mu=nan(numel(keys),1); sd=mu; for k=1:numel(keys), r=S(string(S.model)==m & string(S.feature_set)==keys(k),:); if ~isempty(r), mu(k)=r.R2_mean(1); sd(k)=r.R2_sd(1); end, end
    errorbar(1:numel(keys),mu,sd,'-o','DisplayName',m,'LineWidth',1.2,'CapSize',8);
end
yline(0,'--'); xlim([.7 numel(keys)+.3]); xticks(1:numel(keys)); xticklabels(keys); xtickangle(20); ylabel('Dış fold test R^2 (ortalama ± SD)'); xlabel('Kontrollü özellik aşaması'); title('Kontrollü Ablation'); legend('Location','best'); grid on; hold off;
end
function S=finalPerformanceSummary(F,seed)
if isempty(F), S=table(); return; end
F.model=string(F.model); models=unique(F.model,'stable'); rows={};
for i=1:numel(models)
    P=F(F.model==models(i),:); rci=bootstrapMeanCI(P.test_r2,5000,seed); mci=bootstrapMeanCI(P.test_rmse,5000,seed+1);
    rows(end+1,:)={models(i),mean(P.test_r2),std(P.test_r2,0),rci(1),rci(2),mean(P.test_rmse),std(P.test_rmse,0),mci(1),mci(2),mean(P.test_mae),std(P.test_mae,0),mean(P.train_r2),mean(P.train_r2)-mean(P.test_r2),numel(unique(P.outer_fold)),sum(P.elapsed_seconds)};
end
S=table(string(vertcat(rows{:,1})),cell2mat(rows(:,2)),cell2mat(rows(:,3)),cell2mat(rows(:,4)),cell2mat(rows(:,5)),cell2mat(rows(:,6)),cell2mat(rows(:,7)),cell2mat(rows(:,8)),cell2mat(rows(:,9)),cell2mat(rows(:,10)),cell2mat(rows(:,11)),cell2mat(rows(:,12)),cell2mat(rows(:,13)),cell2mat(rows(:,14)),cell2mat(rows(:,15)), ...
'VariableNames',{'Model','R2_mean','R2_sd','R2_CI_low','R2_CI_high','RMSE_mean','RMSE_sd','RMSE_CI_low','RMSE_CI_high','MAE_mean','MAE_sd','Train_R2_mean','Overfit_gap','Fold_count','Total_time_seconds'}); S=sortrows(S,'R2_mean','descend');
end
function ci=bootstrapMeanCI(v,nBoot,seed)
v=double(v(:)); rng(seed,'twister'); B=zeros(nBoot,1); n=numel(v); for i=1:nBoot, B(i)=mean(v(randi(n,n,1))); end; ci=prctile(B,[2.5 97.5]);
end
function [friedmanResult,averageRanks,posthoc]=finalStatisticalTests(F)
F.model=string(F.model); models=unique(F.model,'stable'); folds=unique(F.outer_fold); score=nan(numel(folds),numel(models));
for i=1:numel(folds), for j=1:numel(models), q=F(F.outer_fold==folds(i)&F.model==models(j),:); if ~isempty(q), score(i,j)=q.test_r2(1); end, end, end
score=score(all(isfinite(score),2),:); if size(score,1)<3, error('Friedman testi için yeterli eşleştirilmiş fold yok.'); end
[p,tbl]=friedman(score,1,'off'); chi=tbl{2,5}; friedmanResult=table("test_r2",size(score,1),size(score,2),chi,p,p<.05,'VariableNames',{'metric','n_blocks','n_models','friedman_chi_square','p_value','significant_0_05'});
ranks=zeros(size(score)); for i=1:size(score,1), ranks(i,:)=tiedrank(-score(i,:)); end; ar=mean(ranks,1)'; averageRanks=table(models,ar,'VariableNames',{'Model','average_rank'}); averageRanks=sortrows(averageRanks,'average_rank','ascend');
rows={}; for a=1:numel(models)-1, for b=a+1:numel(models), d=score(:,a)-score(:,b); if all(abs(d)<1e-12), stat=0; pp=1; else, [pp,~,st]=signrank(score(:,a),score(:,b)); stat=st.signedrank; end; rows(end+1,:)={models(a),models(b),mean(d),median(d),rankBiserial(d),stat,pp,numel(d)}; end, end
posthoc=table(string(vertcat(rows{:,1})),string(vertcat(rows{:,2})),cell2mat(rows(:,3)),cell2mat(rows(:,4)),cell2mat(rows(:,5)),cell2mat(rows(:,6)),cell2mat(rows(:,7)),cell2mat(rows(:,8)), ...
'VariableNames',{'Model_A','Model_B','Mean_R2_A_minus_B','Median_R2_A_minus_B','Rank_biserial_A_vs_B','Wilcoxon_statistic','p_raw','n_paired_folds'}); posthoc.p_holm=holmAdjust(posthoc.p_raw); posthoc.significant_holm_0_05=posthoc.p_holm<.05; posthoc=sortrows(posthoc,{'p_holm','p_raw'},{'ascend','ascend'});
end
function r=rankBiserial(d)
d=d(:); d=d(abs(d)>1e-12); if isempty(d), r=0; return; end; rk=tiedrank(abs(d)); pos=sum(rk(d>0)); neg=sum(rk(d<0)); r=(pos-neg)/(pos+neg);
end
function adj=holmAdjust(p)
p=p(:); m=numel(p); [ps,ord]=sort(p); a=zeros(m,1); running=0; for i=1:m, running=max(running,(m-i+1)*ps(i)); a(i)=min(1,running); end; adj=zeros(m,1); adj(ord)=a;
end
function f=plotFinalR2Boxplot(F,S,featureKey)
order=string(S.Model); cats=categorical(string(F.model),order,'Ordinal',true); f=figure('Name','Nested CV Model Karşılaştırması'); boxchart(cats,F.test_r2,'MarkerStyle','.'); yline(0,'--'); ylabel('Dış fold test R^2'); xlabel('Model'); title('Nested CV Model Karşılaştırması — '+string(featureKey)); xtickangle(35); grid on;
end
function f=plotActualPredicted(P,bestModel)
Q=P(string(P.model)==bestModel,:); lo=min([Q.y_true;Q.y_pred]); hi=max([Q.y_true;Q.y_pred]); f=figure('Name','Gerçek-Tahmin'); scatter(Q.y_true,Q.y_pred,36,'filled','MarkerFaceAlpha',.65); hold on; plot([lo hi],[lo hi],'--','LineWidth',1.2); hold off; xlabel('Gerçek HI'); ylabel('Dış fold tahmini HI'); title('Gerçek–Tahmin: '+bestModel); axis square; grid on;
end
function [r2,rmse,mae]=metrics3(y,p), y=y(:); p=p(:); r2=r2score(y,p); rmse=sqrt(mean((y-p).^2)); mae=mean(abs(y-p)); end
function r=r2score(y,p), den=sum((y-mean(y)).^2); if den<eps, r=NaN; else, r=1-sum((y-p).^2)/den; end, end
function v=safeDivide(a,b), v=a./b; v(~isfinite(v)|b==0)=NaN; end
function x=toNumeric(col)
if isnumeric(col)||islogical(col), x=double(col); else, s=strip(string(col)); bad=ismissing(s)|s==""|s=="-"|s==" -"; s(bad)=missing; x=str2double(s); end; x=x(:);
end
function s=columnToStrings(col)
if iscategorical(col), s=string(col); elseif isstring(col), s=col; elseif iscell(col), s=string(col); elseif isnumeric(col)||islogical(col), s=string(col); else, s=string(col); end; s=strip(s(:));
end
function m=isMissingLike(col), if isnumeric(col), m=~isfinite(double(col)); else, m=isMissingString(columnToStrings(col)); end, end
function m=isMissingString(s), m=ismissing(s)|s==""|s=="-"|s==" -"|lower(s)=="nan"|lower(s)=="<missing>"; end
function lev=sortCategories(lev)
lev=string(lev(:)); nums=str2double(lev); if ~isempty(lev)&&all(isfinite(nums)), [~,ix]=sort(nums); lev=lev(ix); else, lev=sort(lev); end
end
function m=mostFrequentCategory(s)
s=string(s(:)); if isempty(s), m=""; return; end; lev=sortCategories(unique(s)); counts=zeros(numel(lev),1); for i=1:numel(lev), counts(i)=sum(s==lev(i)); end; [~,ix]=max(counts); m=lev(ix);
end
function seed=stableNameSeed(txt), c=double(char(string(txt))); seed=mod(sum((1:numel(c)).*c),10000); end
function [manifestPath,zipPath]=saveManifestAndArchive(cfg,rawExclusions,groupColumn,resultTag,fingerprint)
v=ver; env=struct2table(v); writetable(env,fullfile(cfg.OUTPUT_DIR,"environment_versions_matlab.csv")); pyStatus=checkPythonBridge(); writetable(pyStatus,fullfile(cfg.OUTPUT_DIR,"environment_versions_python.csv"));
manifest=struct('CODE_VERSION',cfg.CODE_VERSION,'MATLAB_RELEASE',string(version('-release')),'FAST_MODE',cfg.FAST_MODE,'RESULT_TAG',resultTag,'CONFIG_FINGERPRINT',fingerprint,'SEED',cfg.SEED,'TARGET',cfg.TARGET,'DATA_PATH',cfg.DATA_PATH,'LEAKAGE_POLICY',cfg.LEAKAGE_POLICY,'RAW_EXCLUSIONS',rawExclusions,'GROUP_COLUMN',groupColumn,'FINAL_FEATURE_SET_KEY',cfg.FINAL_FEATURE_SET_KEY,'OUTER_SPLITS',cfg.OUTER_SPLITS,'OUTER_REPEATS',cfg.OUTER_REPEATS,'INNER_SPLITS',cfg.INNER_SPLITS,'CORE_MODELS',cfg.CORE_MODELS,'ABLATION_MODELS',cfg.ABLATION_MODELS);
manifestPath=fullfile(cfg.OUTPUT_DIR,"run_manifest_"+resultTag+".json"); fid=fopen(manifestPath,'w','n','UTF-8'); assert(fid>0,'Manifest dosyası açılamadı.'); cleanup=onCleanup(@()fclose(fid)); fprintf(fid,'%s',jsonencode(manifest,'PrettyPrint',true)); clear cleanup;
zipPath=fullfile(cfg.WORK_DIR,cfg.OUTPUT_DIR_NAME+"_"+resultTag+".zip"); if isfile(zipPath), delete(zipPath); end; zip(zipPath,"*",cfg.OUTPUT_DIR);
end