import SwiftUI
import Amplify

struct ContentView: View {

    // MARK: - Input fields
    @State private var name = ""
    @State private var description = ""

    // MARK: - Data
    @State private var todos: [Todo] = []

    var body: some View {
        VStack(spacing: 16) {

            Text("Add Item")
                .font(.title)

            TextField("Name", text: $name)
                .textFieldStyle(.roundedBorder)
                .padding(.horizontal)

            TextField("Description", text: $description)
                .textFieldStyle(.roundedBorder)
                .padding(.horizontal)

            Button("Save") {
                createTodo()
            }
            .padding()
            .frame(maxWidth: .infinity)
            .background(Color.blue)
            .foregroundColor(.white)
            .cornerRadius(8)
            .padding(.horizontal)

            Button("Refresh") {
                fetchTodos()
            }
            .padding()
            .frame(maxWidth: .infinity)
            .background(Color.green)
            .foregroundColor(.white)
            .cornerRadius(8)
            .padding(.horizontal)

            // MARK: - Top 5 list from GraphQL
            List(todos.prefix(5), id: \.id) { todo in
                VStack(alignment: .leading, spacing: 4) {
                    Text(todo.name)
                        .font(.headline)

                    Text(todo.description ?? "")
                        .font(.subheadline)
                        .foregroundColor(.gray)
                }
            }
        }
        .padding()
        .onAppear {
            fetchTodos()
        }
    }

    // MARK: - CREATE
    func createTodo() {

        let request = GraphQLRequest<Todo>(
            document: """
            mutation CreateTodo($input: CreateTodoInput!) {
                createTodo(input: $input) {
                    id
                    name
                    description
                    createdAt
                    updatedAt
                }
            }
            """,
            variables: [
                "input": [
                    "name": name,
                    "description": description
                ]
            ],
            responseType: Todo.self
        )

        Task {
            do {
                let result = try await Amplify.API.mutate(request: request)
                print("Saved:", result)
                fetchTodos()
            } catch {
                print("Create error:", error)
            }
        }
    }

    // MARK: - FETCH
    func fetchTodos() {

        let request = GraphQLRequest<ListTodoResponse>(
            document: """
            query ListTodos {
                listTodos {
                    items {
                        id
                        name
                        description
                        createdAt
                    }
                }
            }
            """,
            responseType: ListTodoResponse.self
        )

        Task {
            do {
                let result = try await Amplify.API.query(request: request)

                switch result {
                case .success(let data):
                    await MainActor.run {
                        self.todos = data.listTodos.items.compactMap { $0 }
                    }

                case .failure(let error):
                    print("Query failed:", error)
                }

            } catch {
                print("Network error:", error)
            }
        }
    }
}

struct ListTodoResponse: Sendable {
    let listTodos: TodoItems
}

extension ListTodoResponse: Decodable {
    nonisolated init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        listTodos = try container.decode(TodoItems.self, forKey: .listTodos)
    }

    enum CodingKeys: String, CodingKey {
        case listTodos
    }
}

struct TodoItems: Sendable {
    let items: [Todo?]
}

extension TodoItems: Decodable {
    nonisolated init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        items = try container.decode([Todo?].self, forKey: .items)
    }

    enum CodingKeys: String, CodingKey {
        case items
    }
}
