//
//  EDENApp.swift
//  EDEN
//
//  Created by Alana Kumar on 17/4/2026.
//

import SwiftUI

@main
struct EDENApp: App {
    @StateObject var movieVM = MovieViewModel()
    @StateObject var showVM = TVViewModel()
    var body: some Scene {
        WindowGroup {
            NavigationStack{
                ContentView()
            }
            .environmentObject(movieVM)
            
        }
    }
}
