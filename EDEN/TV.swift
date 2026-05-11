//
//  TV.swift
//  EDEN
//
//  Created by Alana Kumar on 1/5/2026.
//

import Foundation

struct TV: Identifiable, Decodable {
    let id: Int
    let name: String
    let poster_path: String
    let overview: String
}

struct TVResponse: Decodable {
    let results: [TV]
}
