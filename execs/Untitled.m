close all

inp_files = dir('input*.txt');

% Number of input and output files
num_files_inp =length(inp_files);


for ind = 1:num_files_inp
    
    % Construct input and output file names using the loop variable
    input_file_name = sprintf('input%d.txt', ind);
    output_file_prefix = sprintf('output%d_', ind);
    
% %     Get the list of all input and output files in the directory
%     input_files = dir('input2.txt');
%     output_files = dir('output2_*.txt');
    % Get the list of all input and output files in the directory
    input_files = dir(input_file_name);
    output_files = dir([output_file_prefix, '*.txt']);

    % Number of input and output files
    num_files =length(output_files);

    % Create a new figure for visualizing both graphs
    figure;

    alpha_num_arr = zeros (size(output_files));

    for file_num = 1:num_files

        % Read the input graph data from input.txt
        file_name_input = input_file_name;
        data_input = dlmread(file_name_input);

        % Extract the edges from the input data
        edges_input = data_input(:, 1:2);

        % Get the unique vertices in the graph
        vertices = unique(edges_input(:));

        % Create a graph from the edges and label the nodes
        G_input = graph(edges_input(:, 1)+ 1 , edges_input(:, 2) +1 );
        G_input.Nodes.Name = arrayfun(@(x) num2str(x), vertices, 'UniformOutput', false);


        % Read the graph embedding data from output.txt
        file_name_output = output_files(file_num).name;
        data_output = dlmread(file_name_output);

        parts = strsplit(file_name_output, '_'); % Split by underscores
        output_num = str2double(parts{1}(7:end)); % Extract number after "output"
        pos_num = str2double(parts{2}(4:end));    % Extract number after "pos"
        alpha_num = str2double(parts{3}(6:end));  % Extract number after "alpha"
        alpha_num_arr = vertcat(alpha_num_arr, alpha_num);
        % Extract the wedge number by removing ".txt" from the last part
        wedge_str = parts{4};
        wedge_num = str2double(wedge_str(6:end-4)); % Remove ".txt"


        % Extract the embedding data and dimensions from the output data
        num_vertices = data_output(1, 1);
        dimension = data_output(1, 2);
        embedding_data = data_output(2:end, 2:end); % Read only the embedding data

        % Create a tuple containing the original vertices and their embeddings
        vertices = unique(edges_input(:));
        vertex_embedding_tuple = [vertices, embedding_data];

        % Sort the tuple based on the second column (embedding values)
        [~, sorted_indices] = sort(vertex_embedding_tuple(1:end, 2));
        sorted_vertex_embedding_tuple = vertex_embedding_tuple(sorted_indices, :);

        % Get the sorted vertices and corresponding embedding values
        sorted_vertices = sorted_vertex_embedding_tuple(:, 1);
        sorted_embedding_data = sorted_vertex_embedding_tuple(:, 2:end);

        % Subplot for the original graph visualization
        subplot(6, 5, 1);
        p_input = plot(G_input, 'Layout', 'force', 'EdgeColor', 'b', 'MarkerSize', 3, 'NodeColor', 'r', 'NodeLabel', G_input.Nodes.Name);
        title(['Original Graph Visualization (', file_name_input, ')']);

        %--------------------------------------------------------
        % ADJACENCY MATRIX ORIGINAL SPY 1
        % Subplot for the adjacency matrix heatmap
        subplot(6, 5, 2);
        % adjacency_matrix = (adjacency(G_input));

        % Initialize the adjacency matrix with zeros MANUALLY CREATE
        adjacency_matrix = zeros(length(vertices));


        % Set the values in the adjacency matrix based on the edges (undirected)
        for i = 1:size(edges_input, 1)
            vertex1 = edges_input(i, 1) ;
            vertex2 = edges_input(i, 2) ;
            adjacency_matrix(vertex1 + 1, vertex2 + 1) = 1;
            adjacency_matrix(vertex2 + 1, vertex1 + 1) = 1; % Ensure symmetry
        end

         % Subplot for the sparsity pattern of the adjacency matrix
        spy(adjacency_matrix);    
        title(['Adjacency ORIGINAL (', file_name_input, ')']);
        xlabel('Vertex Index');
        ylabel('Vertex Index');

        % Customize colors for the spy plot
        colormap(gca, [1, 1, 1; 0, 0, 0]); % White for zeros, Black for non-zeros


          % Keep the x and y tick labels as the sorted vertex indices
        xticks(1:size(adjacency_matrix, 1));
        yticks(1:size(adjacency_matrix, 1));
        xticklabels(num2str(vertices(1:end) ));
        yticklabels(num2str(vertices(1:end) ));

       % yticklabels(num2str(vertices(end:-1:1))); % Reverse the order
        xlabel('Original Vertex Index');
        ylabel('Original Vertex Index');

        %--------------------------------------------------------



        subplot(6, 5, 3);
        % Calculate the Reverse Cuthill-McKee (RCM) ordering
        rcm_order = symrcm(adjacency_matrix);
        spy(adjacency_matrix(rcm_order, rcm_order));
        title(['Adjacency RCM']);
        xlabel('RCM Vertex Index');
        ylabel('RCM Vertex Index');

        % Customize colors for the spy plot
        colormap(gca, [1, 1, 1; 0, 0, 0]); % White for zeros, Black for non-zeros

        % Set tick positions and labels for the RCM-ordered vertex indices
        tick_positions = 1:size(adjacency_matrix, 1);
        tick_labels = num2str(rcm_order' - 1);

        % Adjust tick positions and labels for better readability 
        % For example, to show every 10th vertex index
        % tick_positions = 1:10:size(adjacency_matrix, 1);
        % tick_labels = num2str(rcm_order(tick_positions)');

        xticks(tick_positions);
        yticks(tick_positions);
        xticklabels(tick_labels);
        yticklabels(tick_labels);

        xtickangle(45); % Rotate tick labels for better visibility 


        subplot(6, 5, 4);
         % Calculate the Approximate Minimum Degree (AMD) ordering
        amd_order = amd(adjacency_matrix);
        spy(adjacency_matrix(amd_order, amd_order));
        title(['Adjacency Matrix AMD']);
        xlabel('AMD Vertex Index');
        ylabel('AMD Vertex Index');

          % Customize colors for the spy plot
        colormap(gca, [1, 1, 1; 0, 0, 0]); % White for zeros, Black for non-zeros

        % Set tick positions and labels for the RCM-ordered vertex indices
        tick_positions = 1:size(adjacency_matrix, 1);
        tick_labels = num2str(amd_order' - 1);

        % Adjust tick positions and labels for better readability
        % For example, to show every 10th vertex index
        % tick_positions = 1:10:size(adjacency_matrix, 1);
        % tick_labels = num2str(rcm_order(tick_positions)');

        xticks(tick_positions);
        yticks(tick_positions);
        xticklabels(tick_labels);
        yticklabels(tick_labels);

        xtickangle(45); % Rotate tick labels for better visibility

        %--------------------------------------------------------
%         % Subplot for embedding values
% 
%         subplot(5, 4, 4+(2*file_num-1));
%         plot(sorted_vertex_embedding_tuple(:, 2), 'LineWidth', 2);
% 
%         % Create the dynamic title string using the extracted values
%         dynamic_title = sprintf('Out:%d, Pos:%d, a:%.2f, w:%d', ...
%                                 output_num, pos_num, alpha_num, wedge_num);
% 
%         % existing title
%         existing_title = 'Sorted Embed';
% 
%         % Combine the existing title with the dynamic title
%         combined_title = [existing_title, '-', dynamic_title];
% 
%         title(combined_title);

        sorted_vertex_indices = sorted_vertex_embedding_tuple(:, 1);
        adjacency_matrix_sorted = adjacency_matrix(sorted_indices, sorted_indices);


        % Subplot for he sparsity pattern of the SORTED adjacency matrix
        subplot(6, 5, (4+file_num));
        spy(adjacency_matrix_sorted);

        % Create the dynamic title string using the extracted values
        dynamic_title = sprintf('Out:%d, Pos:%d, a:%.2f, w:%d',...
                                output_num, pos_num, alpha_num, wedge_num);

        % existing title
        existing_title = 'GOSH';

        % Combine the existing title with the dynamic title
        combined_title = [existing_title, '-', dynamic_title];

        title(combined_title);

        xlabel('V Ind');
        ylabel('V Ind');

        % Customize colors for the spy plot
        colormap(gca, [1, 1, 1; 0, 0, 0]); % White for zeros, Black for non-zeros

        % Keep the x and y tick labels as the sorted vertex indices
        xticks(1:size(adjacency_matrix_sorted, 1));
        yticks(1:size(adjacency_matrix_sorted, 1));
        xticklabels(num2str(sorted_vertex_indices));
        yticklabels(num2str(sorted_vertex_indices));


        xlabel('V Ind');
        ylabel('V Ind');

    end
end

