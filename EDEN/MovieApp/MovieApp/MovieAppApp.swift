//
//  MovieAppApp.swift
//  MovieApp
//
//  Created by Eden Hallett on 20/4/2026.
//
import SwiftUI
import Amplify
import AWSAPIPlugin
import AWSCognitoAuthPlugin

@main
struct MovieAppApp: App {

    init() {
        configureAmplify()
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

func configureAmplify() {
    do {
        try Amplify.add(plugin: AWSAPIPlugin())
        try Amplify.add(plugin: AWSCognitoAuthPlugin())

        try Amplify.configure()
        print("Amplify configured successfully")
    } catch {
        print("Failed to configure Amplify:", error)
    }
}
