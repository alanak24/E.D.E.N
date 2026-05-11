//
//  TVViewModel.swift
//  EDEN
//
//  Created by Alana Kumar on 1/5/2026.
//

import Foundation

import SwiftUI

class TVViewModel: ObservableObject {
    @Published var shows: [TV] = []
    @Published var preferences: [Int: UserTVPreference] = [:]


    func fetchShows() {
        guard let url = URL(string:
            "https://api.themoviedb.org/3/tv/top_rated?api_key=513b40cf1f53d69edefdd4c32a9b72b2"
        ) else { return }

        URLSession.shared.dataTask(with: url) { data, _, _ in
            guard let data = data else { return }

            do {
                let decoded = try JSONDecoder().decode(TVResponse.self, from: data)

                DispatchQueue.main.async {
                    self.shows = Array(decoded.results.prefix(20))
                }
            } catch {
                print("Error decoding TV:", error)
            }
        }.resume()
    }
    
    func toggleLike(for showId: Int) {
        if preferences[showId] != nil {
            preferences[showId]?.isLiked.toggle()
        } else {
            preferences[showId] = UserTVPreference(
                showId: showId,
                rating: nil,
                isLiked: true
            )
        }
    }

    func setRating(_ value: Double, for showId: Int) {
        if preferences[showId] != nil {
            preferences[showId]?.rating = value
        } else {
            preferences[showId] = UserTVPreference(
                showId: showId,
                rating: value,
                isLiked: false
            )
        }
    }

    func setLike(_ value: Bool, for showId: Int) {
        if preferences[showId] != nil {
            preferences[showId]?.isLiked = value
        } else {
            preferences[showId] = UserTVPreference(
                showId: showId,
                rating: nil,
                isLiked: value
            )
        }
    }
}
