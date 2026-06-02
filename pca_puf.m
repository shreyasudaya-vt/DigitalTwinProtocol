function final_pca()
    %% Single vs Multi Sweep PCA Authentication Framework
    % Generates IDs using PCA from device sweep CSVs (like 1_1.csv ... 1_7.csv).
    % Performs authentication, computes success %, Hamming distances, and writes a
    % Final_Results.md with embedded plots and explanations.
    
    clear; clc; close all;
    
    script_dir = fileparts(mfilename('fullpath'));
    
    % 2. Build absolute paths to your dataset and report folders
    DEVICE_FOLDER = fullfile(script_dir, '01_master_dataset');    
    REPORT_DIR    = fullfile(script_dir, '01_256reports');         
    
    % 3. Double-check to make sure the folder actually exists before proceeding
    if ~exist(DEVICE_FOLDER, 'dir')
        error('Error: The folder "%s" was not found! Check its spelling or location.', DEVICE_FOLDER);
    end
    PREFERRED_REG_INDEX_SINGLE    = 1;
    PREFERRED_AUTH_INDEX_SINGLE   = 2;
    PREFERRED_MULTI_TRAIN_INDICES = 4:6;       % equivalent to list(range(4,7)) -> [4, 5, 6]
    PREFERRED_MULTI_AUTH_INDEX    = 5;
    USE_PHASE     = true;
    ID_BIT_LENGTH = 128;
    START_FREQ    = 10000;
    END_FREQ      = 1000000;
    N_FREQ_POINTS = 500;
    REF_FREQ      = linspace(START_FREQ, END_FREQ, N_FREQ_POINTS);
    % CUSTOM CONTROLS
    EXCLUDED_DEVICES       = {'201', '253', '254', '258', '310'}; % e.g., devices to skip
    HAMMING_AUTH_THRESHOLD = 25;                                  % max acceptable intra-distance
    if ~exist(REPORT_DIR, 'dir')
        mkdir(REPORT_DIR);
    end
    %% ========================================================================
    %% MAIN EXECUTION PIPELINE
    %% ========================================================================
    device_files = collect_device_files(DEVICE_FOLDER);
    dev_keys = sorted_keys(device_files);
    fprintf('--> Devices Found: %s\n', strjoin(dev_keys, ', '));
    % -------------------------------------------------------------------------
    % SINGLE SWEEP PCA CONFIGURATION
    % -------------------------------------------------------------------------
    fprintf('\n--> Executing Single-Sweep Registration Phase...\n');
    [order, bin_ids, model] = build_single_sweep_ids(device_files, PREFERRED_REG_INDEX_SINGLE, REF_FREQ, USE_PHASE, ID_BIT_LENGTH);
    
    
    fprintf('--> Authenticating Single-Sweep Projections...\n');
    [single_results, flags_s, intra_s, inter_s] = authenticate_files(...
        model, bin_ids, device_files, PREFERRED_AUTH_INDEX_SINGLE, ...
        HAMMING_AUTH_THRESHOLD, EXCLUDED_DEVICES ...
    );
    if ~isempty(EXCLUDED_DEVICES)
        fprintf('    Excluded devices: %s\n', strjoin(EXCLUDED_DEVICES, ', '));
    end
    % Save single-sweep metrics CSV
    single_table = results2table(single_results);
    writetable(single_table, fullfile(REPORT_DIR, 'single_sweep_metrics.csv'));
    
    % Plot Combined Histogram Metrics
    plot_combined_pdf(...
        intra_s, inter_s, ...
        'Hamming Distance Distribution (Single-Sweep PCA)', ...
        fullfile(REPORT_DIR, 'single_combined.png') ...
    );
    % Calculate single-sweep authentication rate
    total_reg_s = length(order);
    success_s = 0;
    for i = 1:length(order)
        d = order{i};
        if isKey(flags_s, d) && all(flags_s(d))
            success_s = success_s + 1;
        end
    end
    if total_reg_s > 0, rate_s = (success_s / total_reg_s) * 100; else, rate_s = 0; end
    % -------------------------------------------------------------------------
    % MULTI SWEEP PCA CONFIGURATION
    % -------------------------------------------------------------------------
    fprintf('\n--> Executing Multi-Sweep Registration Phase...\n');
    [bin_ids_m, model_m] = build_multi_sweep_ids(device_files, PREFERRED_MULTI_TRAIN_INDICES, REF_FREQ, USE_PHASE, ID_BIT_LENGTH);
    fprintf('--> Authenticating Multi-Sweep Projections...\n');
    [multi_results, flags_m, intra_m, inter_m] = authenticate_files(...
        model_m, bin_ids_m, device_files, PREFERRED_MULTI_AUTH_INDEX, ...
        HAMMING_AUTH_THRESHOLD, EXCLUDED_DEVICES ...
    );
    % Save multi-sweep metrics CSV
    multi_table = results2table(multi_results);
    writetable(multi_table, fullfile(REPORT_DIR, 'multi_sweep_metrics.csv'));
    

    % Calculate multi-sweep authentication rate
    m_keys = sorted_keys(bin_ids_m);
    total_reg_m = length(m_keys);
    success_m = 0;
    for i = 1:length(m_keys)
        d = m_keys{i};
        if isKey(flags_m, d) && all(flags_m(d))
            success_m = success_m + 1;
        end
    end
    if total_reg_m > 0, rate_m = (success_m / total_reg_m) * 100; else, rate_m = 0; end
    % -------------------------------------------------------------------------
    % EXPORT DATA TABLES & SUMMARY REPORT
    % -------------------------------------------------------------------------
    Scenario = {'Single'; 'Multi'};
    Registered = [total_reg_s; total_reg_m];
    Success = [success_s; success_m];
    Rate_Pct = [rate_s; rate_m];
    MeanIntra = [mean(intra_s); mean(intra_m)];
    comp_table = table(Scenario, Registered, Success, Rate_Pct, MeanIntra, ...
                       'VariableNames', {'Scenario', 'Registered', 'Success', 'Rate_Pct', 'MeanIntra'});
    writetable(comp_table, fullfile(REPORT_DIR, 'comparison_summary.csv'));
    % Export the debug records
    export_device_debug_data(REPORT_DIR, bin_ids, bin_ids_m, single_results, multi_results);
    % Generate Markdown Document Report
    fid = fopen(fullfile(REPORT_DIR, 'Final_Results.md'), 'w');
    if fid ~= -1
        fprintf(fid, '# Final Results Report\n\n');
        fprintf(fid, '## What are intra vs inter Hamming graphs?\n');
        fprintf(fid, '- **Intra-device distances**: Hamming distance between a regenerated ID (at authentication) and its *own* registered ID.\n');
        fprintf(fid, '  These show how stable/reproducible each device''s ID is over time.\n');
        fprintf(fid, '- **Inter-device distances**: Hamming distances between a regenerated ID and *all other devices''* registered IDs.\n');
        fprintf(fid, '  These show how well-separated the devices are (uniqueness).\n\n');
        fprintf(fid, 'Ideally: intra distances are low (close to 0), while inter distances are high (close to half the ID length).\n\n');
        fprintf(fid, '## Single-sweep (5 -> 6)\n');
        fprintf(fid, '- Devices registered: **%d**\n- Successfully authenticated: **%d**\n', total_reg_s, success_s);
        fprintf(fid, '- Success rate: **%.2f%%**\n- Mean intra Hamming: **%.2f**\n\n', rate_s, mean(intra_s));
        fprintf(fid, '### Plots\n');
        fprintf(fid, '![Single Combined](single_combined.png)\n\n');
        fprintf(fid, '## Multi-sweep (1..6 -> 7)\n');
        fprintf(fid, '- Devices registered: **%d**\n- Successfully authenticated: **%d**\n', total_reg_m, success_m);
        fprintf(fid, '- Success rate: **%.2f%%**\n- Mean intra Hamming: **%.2f**\n\n', rate_m, mean(intra_m));
        fprintf(fid, '### Plots\n');
        fprintf(fid, '![Multi Combined](multi_combined.png)\n\n');
        fclose(fid);
    end
    fprintf('\n--> Processing complete! Reports saved to %s\n', REPORT_DIR);
end
%% ========================================================================
%% LOCAL UTILITY FUNCTIONS
%% ========================================================================
function [freq, phase, imp] = load_sweep_data(path, use_phase)
    % Robust custom parser that targets data despite headers or comments
    opts = detectImportOptions(path, 'FileType', 'text');
    
    % Use try-catch blocks for compatibility with older MATLAB versions
    try
        opts.VariableNamingRule = 'preserve';
    catch
        % Older MATLABs will auto-sanitize names (e.g. 'Trace M' -> 'Trace_M')
    end
    
    try
        opts.VariableNamingOptions.ReservedWordsAction = 'none';
    catch
    end
    
    tbl = [];
    % Skip variants for the header rows (adjust based on CSV format)
    skip_variants = [32, 33, 1, 0];
    for s = skip_variants
        try
            opts.DataLines = [s + 1, Inf];
            tbl = readtable(path, opts);
            if ~isempty(tbl) && size(tbl, 2) >= 2
                break;
            end
        catch
            continue;
        end
    end
    
    if isempty(tbl)
        error('Could not parse layout structure for CSV: %s', path);
    end
    
    col_names_lower = lower(tbl.Properties.VariableNames);
    
    % Track active column positions (fuzzy matching works on older MATLABs too)
    freq_idx = 1;
    for j = 1:length(col_names_lower)
        if contains(col_names_lower{j}, 'frequency') || contains(col_names_lower{j}, 'freq') || strcmp(col_names_lower{j}, 'f')
            freq_idx = j;
            break;
        end
    end
    
    imp_idx = 2;
    for j = 1:length(col_names_lower)
        if contains(col_names_lower{j}, 'trace |z|') || contains(col_names_lower{j}, 'impedance') || ...
           contains(col_names_lower{j}, '|z|') || contains(col_names_lower{j}, 'imp') || ...
           strcmp(col_names_lower{j}, 'z') || contains(col_names_lower{j}, 'trace') || contains(col_names_lower{j}, 'm_db_')
            imp_idx = j;
            break;
        end
    end
    
    phase_idx = [];
    if use_phase
        for j = 1:length(col_names_lower)
            if contains(col_names_lower{j}, 'trace th') || contains(col_names_lower{j}, 'phase') || ...
               contains(col_names_lower{j}, 'angle') || strcmp(col_names_lower{j}, 'th')
                phase_idx = j;
                break;
            end
        end
    end
    
    freq = tbl{:, freq_idx};
    imp = tbl{:, imp_idx};
    if ~isempty(phase_idx)
        phase = tbl{:, phase_idx};
    else
        phase = [];
    end
end
function vec = load_sweep_vector(filepath, ref_freq, use_phase)
    [freq, phase, imp] = load_sweep_data(filepath, use_phase);
    
    if freq(1) > freq(end)
        freq = flipud(freq);
        imp = flipud(imp);
        if ~isempty(phase), phase = flipud(phase); end
    end
    
    % Interpolate parameters (clamped edges to match Python np.interp)
    imp_interp = interp1(freq, imp, ref_freq, 'linear', 'extrap');
    imp_interp(ref_freq < freq(1)) = imp(1);
    imp_interp(ref_freq > freq(end)) = imp(end);
    
    if use_phase
        if isempty(phase)
            phase_interp = zeros(size(ref_freq));
        else
            phase_interp = interp1(freq, phase, ref_freq, 'linear', 'extrap');
            phase_interp(ref_freq < freq(1)) = phase(1);
            phase_interp(ref_freq > freq(end)) = phase(end);
        end
        vec = [phase_interp(:)', imp_interp(:)'];
    else
        vec = imp_interp(:)';
    end
    
    %% OPTIONAL VISUALIZATION FOR USER LOG SPECTRUM CHECK
    % if you want to look at the curves using log scale like real devices:
    % figure; semilogy(ref_freq, imp_interp); grid on;
end
function device_files = collect_device_files(folder)
    device_files = containers.Map();
    file_list = dir(fullfile(folder, '*.csv'));
    
    for k = 1:length(file_list)
        fname = file_list(k).name;
        [~, name, ~] = fileparts(fname);
        if ~contains(name, '_'), continue; end
        
        idx_underscore = find(name == '_', 1, 'last');
        prefix = name(1:idx_underscore-1);
        idx_str = name(idx_underscore+1:end);
        idx_int = str2double(idx_str);
        if isnan(idx_int), continue; end
        
        if ~isKey(device_files, prefix)
            device_files(prefix) = containers.Map('KeyType', 'int32', 'ValueType', 'char');
        end
        sub_map = device_files(prefix);
        sub_map(int32(idx_int)) = fullfile(folder, fname);
    end
end
function n_comp = choose_pca_components(X_rows, X_cols, desired_bits)
    n_comp = max([1, min([desired_bits, X_rows, X_cols])]);
end
function [device_order, binary_ids, model] = build_single_sweep_ids(device_files, reg_index, ref_freq, use_phase, desired_bits)
    dev_keys = sorted_keys(device_files);
    train_vecs = [];
    device_order = {};
    
    for i = 1:length(dev_keys)
        dev = dev_keys{i};
        sub_map = device_files(dev);
        if isKey(sub_map, int32(reg_index))
            vec = load_sweep_vector(sub_map(int32(reg_index)), ref_freq, use_phase);
            train_vecs = [train_vecs; vec(:)'];
            device_order{end+1} = dev; %#ok<AGROW>
        end
    end
    
    if isempty(train_vecs)
        error('No verification sweep files found at index %d', reg_index);
    end
    
    [n_samples, n_features] = size(train_vecs);
    n_comp = choose_pca_components(n_samples, n_features, desired_bits);
    
    % StandardScaler: biased variance configuration (divisor=N)
    scaler_mean = mean(train_vecs, 1);
    scaler_scale = std(train_vecs, 1, 1);
    scaler_scale(scaler_scale == 0) = 1;
    
    X_scaled = (train_vecs - scaler_mean) ./ scaler_scale;
    
    % FIXED: Added 'Economy', true to bypass slow computation of TSQUARED on redundant data
    [pca_components, Xp, ~, ~, ~, pca_mean] = pca(X_scaled, 'Centered', true, 'NumComponents', n_comp, 'Economy', true);
    
    binary_ids = containers.Map();
    for i = 1:length(device_order)
        binary_ids(device_order{i}) = (Xp(i, 1:n_comp) > 0);
    end
    
    model = struct('scaler_mean', scaler_mean, 'scaler_scale', scaler_scale, ...
                   'pca_components', pca_components, 'pca_mean', pca_mean, ...
                   'ref_freq', ref_freq, 'use_phase', use_phase, 'bit_length', n_comp);
end
function [binary_ids_m, model_m] = build_multi_sweep_ids(device_files, train_indices, ref_freq, use_phase, desired_bits)
    X_multi = [];
    labels_multi = {};
    dev_keys = sorted_keys(device_files);
    
    for i = 1:length(dev_keys)
        dev = dev_keys{i};
        sub_map = device_files(dev);
        for j = 1:length(train_indices)
            idx = train_indices(j);
            if isKey(sub_map, int32(idx))
                vec = load_sweep_vector(sub_map(int32(idx)), ref_freq, use_phase);
                X_multi = [X_multi; vec(:)'];
                labels_multi{end+1} = dev; %#ok<AGROW>
            end
        end
    end
    
    if isempty(X_multi)
        error('No multi-sweep registration data located.');
    end
    
    [n_samples, n_features] = size(X_multi);
    n_comp = choose_pca_components(n_samples, n_features, desired_bits);
    
    scaler_mean = mean(X_multi, 1);
    scaler_scale = std(X_multi, 1, 1);
    scaler_scale(scaler_scale == 0) = 1;
    
    X_scaled = (X_multi - scaler_mean) ./ scaler_scale;
    
    % FIXED: Added 'Economy', true to completely disable the slow rank-deficient math looping
    [pca_components, Xp, ~, ~, ~, pca_mean] = pca(X_scaled, 'Centered', true, 'NumComponents', n_comp, 'Economy', true);
    
    binary_ids_m = containers.Map();
    unique_labels = sorted_unique(labels_multi);
    
    for i = 1:length(unique_labels)
        dev = unique_labels{i};
        idxs = find(strcmp(labels_multi, dev));
        mean_proj = mean(Xp(idxs, 1:n_comp), 1);
        binary_ids_m(dev) = (mean_proj > 0);
    end
    
    model_m = struct('scaler_mean', scaler_mean, 'scaler_scale', scaler_scale, ...
                     'pca_components', pca_components, 'pca_mean', pca_mean, ...
                     'ref_freq', ref_freq, 'use_phase', use_phase, 'bit_length', n_comp);
end
function [results, flags, intra, inter] = authenticate_files(model, binary_ids, device_files, test_index, threshold, excluded_devices)
    dev_keys = sorted_keys(device_files);
    results = {};
    flags = containers.Map();
    intra = []; inter = [];
    
    scaler_mean = model.scaler_mean;
    scaler_scale = model.scaler_scale;
    pca_components = model.pca_components;
    pca_mean = model.pca_mean;
    bit_length = model.bit_length;
    all_reg_keys = sorted_keys(binary_ids);
    
    for i = 1:length(dev_keys)
        dev = dev_keys{i};
        if any(strcmp(excluded_devices, dev)), continue; end
        
        sub_map = device_files(dev);
        if ~isKey(sub_map, int32(test_index)), continue; end
        
        filepath = sub_map(int32(test_index));
        vec = load_sweep_vector(filepath, model.ref_freq, model.use_phase);
        
        % Normalize and Project using stored parameters
        vec_scaled = (vec(:)' - scaler_mean) ./ scaler_scale;
        proj = (vec_scaled - pca_mean) * pca_components;
        gen_bin = (proj(1:bit_length) > 0);
        
        if isKey(binary_ids, dev)
            reg_bin = binary_ids(dev);
            intra_d = sum(gen_bin ~= reg_bin);
            intra = [intra; intra_d]; %#ok<AGROW>
        else
            intra_d = NaN;
        end
        
        % Compute Inter-Device Cross Distances
        for j = 1:length(all_reg_keys)
            o_dev = all_reg_keys{j};
            if ~strcmp(o_dev, dev)
                inter = [inter; sum(gen_bin ~= binary_ids(o_dev))]; %#ok<AGROW>
            end
        end
        
        % Calculate best geometric matching candidate
        best = ''; min_d = Inf;
        for j = 1:length(all_reg_keys)
            o_dev = all_reg_keys{j};
            d = sum(gen_bin ~= binary_ids(o_dev));
            if d < min_d
                min_d = d; best = o_dev;
            end
        end
        
        match = strcmp(best, dev);
        if isnan(intra_d)
            threshold_pass = false;
        else
            threshold_pass = (isempty(threshold) || (intra_d <= threshold));
        end
        success = match && threshold_pass;
        
        [~, name, ext] = fileparts(filepath);
        res_struct = struct('File', [name, ext], 'Expected', dev, 'Predicted', best, ...
                            'Intra_Hamming', intra_d, 'Within_Threshold', threshold_pass, ...
                            'Match', match, 'Authenticated', success);
        results{end+1} = res_struct; %#ok<AGROW>
        
        if ~isKey(flags, dev), flags(dev) = []; end
        flags(dev) = [flags(dev), success];
    end
end
function plot_combined_pdf(intra, inter, title_str, outpath)
    if isempty(intra) || isempty(inter), return; end
    
    fig = figure('Position', [100, 100, 600, 400], 'Visible', 'off');
    hold on;
    
    max_val = max([max(inter), max(intra)]);
    bins = linspace(0, max_val, 25);
    
    % Blue = Inter-HD, Red = Intra-HD
    histogram(inter, bins, 'Normalization', 'pdf', 'FaceColor', [31, 119, 180]/255, ...
              'FaceAlpha', 0.6, 'EdgeColor', 'black', 'LineWidth', 0.8, 'DisplayName', 'Inter-HD');
    histogram(intra, bins, 'Normalization', 'pdf', 'FaceColor', [214, 39, 40]/255, ...
              'FaceAlpha', 0.6, 'EdgeColor', 'black', 'LineWidth', 0.8, 'DisplayName', 'Intra-HD');
          
    xline(mean(inter), '--', 'Color', [31, 119, 180]/255, 'LineWidth', 2, ...
          'DisplayName', sprintf('Inter Mean = %.2f', mean(inter)));
    xline(mean(intra), '--', 'Color', [214, 39, 40]/255, 'LineWidth', 2, ...
          'DisplayName', sprintf('Intra Mean = %.2f', mean(intra)));
      
    xlabel('Hamming Distance'); ylabel('Probability Density');
    title(title_str); legend('Location', 'best'); grid on;
    
    saveas(fig, outpath);
    close(fig);
end

function export_device_debug_data(report_dir, single_ids, multi_ids, single_results, multi_results)
    s_keys = sorted_keys(single_ids); m_keys = sorted_keys(multi_ids);
    device_list = sorted_unique([s_keys, m_keys]);
    
    single_map = containers.Map();
    for i = 1:length(single_results)
        single_map(single_results{i}.Expected) = single_results{i};
    end
    multi_map = containers.Map();
    for i = 1:length(multi_results)
        multi_map(multi_results{i}.Expected) = multi_results{i};
    end
    
    Device = {}; Single_ID = {}; Multi_ID = {};
    Single_Intra_Hamming = []; Single_Match = []; Single_Predicted = {};
    Multi_Intra_Hamming = []; Multi_Match = []; Multi_Predicted = {};
    
    for i = 1:length(device_list)
        dev = device_list{i};
        Device{end+1, 1} = dev; %#ok<AGROW>
        
        if isKey(single_ids, dev), Single_ID{end+1, 1} = char('0' + single_ids(dev)); else, Single_ID{end+1, 1} = ''; end
        if isKey(multi_ids, dev), Multi_ID{end+1, 1} = char('0' + multi_ids(dev)); else, Multi_ID{end+1, 1} = ''; end
        
        if isKey(single_map, dev)
            s_res = single_map(dev);
            Single_Intra_Hamming(end+1, 1) = s_res.Intra_Hamming; %#ok<AGROW>
            Single_Match(end+1, 1) = double(s_res.Match); %#ok<AGROW>
            Single_Predicted{end+1, 1} = s_res.Predicted; %#ok<AGROW>
        else
            Single_Intra_Hamming(end+1, 1) = NaN; Single_Match(end+1, 1) = NaN; Single_Predicted{end+1, 1} = '';
        end
        
        if isKey(multi_map, dev)
            m_res = multi_map(dev);
            Multi_Intra_Hamming(end+1, 1) = m_res.Intra_Hamming; %#ok<AGROW>
            Multi_Match(end+1, 1) = double(m_res.Match); %#ok<AGROW>
            Multi_Predicted{end+1, 1} = m_res.Predicted; %#ok<AGROW>
        else
            Multi_Intra_Hamming(end+1, 1) = NaN; Multi_Match(end+1, 1) = NaN; Multi_Predicted{end+1, 1} = '';
        end
    end
    
    T = table(Device, Single_ID, Multi_ID, Single_Intra_Hamming, Single_Match, Single_Predicted, ...
              Multi_Intra_Hamming, Multi_Match, Multi_Predicted);
    writetable(T, fullfile(report_dir, 'device_debug_metrics.csv'));
    fprintf(' Device debug data written to: %s\n', fullfile(report_dir, 'device_debug_metrics.csv'));
end
function T = results2table(results_cell)
    if isempty(results_cell)
        T = table(); return;
    end
    T = struct2table([results_cell{:}], 'AsArray', true);
end
function keys_cell = sorted_keys(map_obj)
    keys_cell = keys(map_obj);
    if ~isempty(keys_cell), keys_cell = sort(keys_cell); end
end
function unq_cell = sorted_unique(cell_array)
    unq_cell = unique(cell_array);
    if ~isempty(unq_cell), unq_cell = sort(unq_cell); end
end