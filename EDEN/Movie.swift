//
//  Movie.swift
//  EDEN
//
//  Created by Alana Kumar on 30/4/2026.
//

import Foundation

struct Movie: Identifiable, Decodable {
    let id: Int
    let title: String
    let poster_path: String
    let overview: String
}

struct MovieResponse: Decodable {
    let results: [Movie]
}
